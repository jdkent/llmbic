"""The schema and migration registry — layer 1 of the reference architecture.

Holds schema versions, field identities, recipes, vocabularies and the
migration graph, and answers the one question the planner needs first: what is
the approved path from the version this record is at to the version we want?

FR-MIG-002/003/004/012 all live in :meth:`Registry.find_path`:

* the graph is directed and acyclic for upgrades;
* a downgrade edge is stored separately and never planned through;
* a path is deterministic given the registry state and the planner policy;
* branches may be registered, but only one approved production path may exist
  between any source/target pair.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .errors import ErrorCode, PathError, RegistryError
from .ids import content_hash
from .migration.spec import Migration
from .recipe import ExtractionRecipe, Vocabulary
from .schema.diff import SchemaDiff, diff_schemas
from .schema.normalized import NormalizedSchema


@dataclass(frozen=True)
class SchemaFamily:
    """A stable name for a related sequence of schemas."""

    name: str
    description: str = ""
    #: Ordered list of version strings, in registration order.
    versions: tuple[str, ...] = ()

    def to_canonical(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "versions": list(self.versions),
        }


@dataclass
class PathPreference:
    """How the planner chooses between valid paths (FR-PLN-007)."""

    #: "cost" | "fidelity" | "hops" | "latency"
    optimise: str = "fidelity"
    #: Allow paths through migrations on a non-main branch.
    branches: tuple[str, ...] = ("main",)
    require_approved: bool = True
    #: Refuse any path containing a step that can send whole documents.
    forbid_full_document: bool = False
    max_hops: int = 32

    def key(self, path: Sequence[Migration]) -> tuple:
        semantic = sum(len(m.semantic_steps()) for m in path)
        worst = max((m.fidelity.rank for m in path), default=0)
        if self.optimise == "cost":
            return (semantic, worst, len(path), tuple(m.id for m in path))
        if self.optimise == "hops":
            return (len(path), worst, semantic, tuple(m.id for m in path))
        if self.optimise == "latency":
            return (sum(len(m.steps) for m in path), len(path), tuple(m.id for m in path))
        # fidelity (default): never take a lossier path to save a model call
        return (worst, semantic, len(path), tuple(m.id for m in path))


class Registry:
    """In-memory registry.  Persistence is a store concern, not this class's."""

    def __init__(self) -> None:
        self._schemas: dict[str, NormalizedSchema] = {}
        self._families: dict[str, SchemaFamily] = {}
        self._migrations: dict[str, Migration] = {}
        self._upgrades: dict[str, list[Migration]] = defaultdict(list)
        self._downgrades: dict[str, list[Migration]] = defaultdict(list)
        self._recipes: dict[str, ExtractionRecipe] = {}
        self._vocabularies: dict[str, Vocabulary] = {}

    # ---- schemas ---------------------------------------------------------
    def register_schema(
        self,
        schema: NormalizedSchema,
        *,
        replace_existing: bool = False,
    ) -> NormalizedSchema:
        """Register an immutable schema version (FR-SCH-001).

        Re-registering the same ref with different content is refused: a
        version identifier that can change meaning is not a version.
        """

        existing = self._schemas.get(schema.ref)
        if existing is not None and not replace_existing:
            if existing.schema_hash() != schema.schema_hash():
                raise RegistryError(
                    f"schema {schema.ref} is already registered with different content; "
                    "register a new version instead",
                    code=ErrorCode.SCHEMA_IMMUTABLE,
                    details={
                        "existing_hash": existing.schema_hash(),
                        "new_hash": schema.schema_hash(),
                    },
                )
            return existing

        self._schemas[schema.ref] = schema
        family = self._families.get(schema.name) or SchemaFamily(schema.name)
        if schema.version not in family.versions:
            family = SchemaFamily(
                family.name, family.description, family.versions + (schema.version,)
            )
        self._families[schema.name] = family
        return schema

    def schema(self, ref: str) -> NormalizedSchema:
        if ref not in self._schemas:
            raise RegistryError(
                f"unknown schema {ref!r}; registered: {sorted(self._schemas)}",
                code=ErrorCode.SCHEMA_NOT_FOUND,
            )
        return self._schemas[ref]

    def has_schema(self, ref: str) -> bool:
        return ref in self._schemas

    def schemas(self) -> list[NormalizedSchema]:
        return [self._schemas[k] for k in sorted(self._schemas)]

    def family(self, name: str) -> SchemaFamily:
        if name not in self._families:
            raise RegistryError(f"unknown schema family {name!r}", code=ErrorCode.SCHEMA_NOT_FOUND)
        return self._families[name]

    def families(self) -> list[SchemaFamily]:
        return [self._families[k] for k in sorted(self._families)]

    def latest(self, family: str) -> NormalizedSchema:
        fam = self.family(family)
        if not fam.versions:
            raise RegistryError(f"family {family!r} has no versions", code=ErrorCode.SCHEMA_NOT_FOUND)
        return self.schema(f"{family}@{fam.versions[-1]}")

    def diff(self, from_ref: str, to_ref: str, *, renames: Mapping[str, str] | None = None) -> SchemaDiff:
        """Diff two registered versions, folding in any declared renames.

        When a migration between the two versions is registered, its ``renames``
        are used automatically — the migration *is* the explicit confirmation
        FR-SCH-004 requires.
        """

        declared = dict(renames or {})
        if not declared:
            for m in self._upgrades.get(from_ref, ()):
                if m.to_schema == to_ref:
                    declared.update(m.renames)
        return diff_schemas(self.schema(from_ref), self.schema(to_ref), renames=declared)

    # ---- recipes and vocabularies ---------------------------------------
    def register_recipe(self, recipe: ExtractionRecipe) -> ExtractionRecipe:
        existing = self._recipes.get(recipe.ref)
        if existing is not None and existing.recipe_hash() != recipe.recipe_hash():
            raise RegistryError(
                f"recipe {recipe.ref} is already registered with different content; "
                "bump the recipe version (that is what invalidates its artifacts)",
                code=ErrorCode.SCHEMA_IMMUTABLE,
            )
        self._recipes[recipe.ref] = recipe
        return recipe

    def recipe(self, ref: str) -> ExtractionRecipe:
        if ref not in self._recipes:
            raise RegistryError(
                f"unknown recipe {ref!r}; registered: {sorted(self._recipes)}",
                code=ErrorCode.MIGRATION_NOT_FOUND,
            )
        return self._recipes[ref]

    def maybe_recipe(self, ref: str | None) -> ExtractionRecipe | None:
        return self._recipes.get(ref) if ref else None

    def recipes(self) -> list[ExtractionRecipe]:
        return [self._recipes[k] for k in sorted(self._recipes)]

    def register_vocabulary(self, vocab: Vocabulary) -> Vocabulary:
        existing = self._vocabularies.get(vocab.ref)
        if existing is not None and existing.vocabulary_hash() != vocab.vocabulary_hash():
            raise RegistryError(
                f"vocabulary {vocab.ref} is already registered with different content",
                code=ErrorCode.SCHEMA_IMMUTABLE,
            )
        self._vocabularies[vocab.ref] = vocab
        return vocab

    def vocabulary(self, ref: str) -> Vocabulary:
        if ref not in self._vocabularies:
            raise RegistryError(
                f"unknown vocabulary {ref!r}", code=ErrorCode.MIGRATION_NOT_FOUND
            )
        return self._vocabularies[ref]

    def maybe_vocabulary(self, ref: str | None) -> Vocabulary | None:
        return self._vocabularies.get(ref) if ref else None

    def vocabularies(self) -> list[Vocabulary]:
        return [self._vocabularies[k] for k in sorted(self._vocabularies)]

    # ---- migrations ------------------------------------------------------
    def register_migration(self, migration: Migration, *, validate: bool = True) -> Migration:
        if migration.id in self._migrations:
            existing = self._migrations[migration.id]
            if existing.migration_hash() != migration.migration_hash():
                raise RegistryError(
                    f"migration {migration.id!r} is already registered with different content; "
                    "a migration identity is immutable",
                    code=ErrorCode.SCHEMA_IMMUTABLE,
                )
            return existing

        if validate:
            problems = self.validate_migration(migration)
            if problems:
                raise RegistryError(
                    f"migration {migration.id!r} is invalid:\n  - " + "\n  - ".join(problems),
                    code=ErrorCode.MIGRATION_INVALID,
                    details={"problems": problems},
                )

        if migration.is_downgrade:
            self._downgrades[migration.from_schema].append(migration)
        else:
            self._assert_no_cycle(migration)
            self._assert_unique_approved(migration)
            self._upgrades[migration.from_schema].append(migration)
            self._upgrades[migration.from_schema].sort(key=lambda m: m.id)

        self._migrations[migration.id] = migration
        return migration

    def migration(self, migration_id: str) -> Migration:
        if migration_id not in self._migrations:
            raise RegistryError(
                f"unknown migration {migration_id!r}", code=ErrorCode.MIGRATION_NOT_FOUND
            )
        return self._migrations[migration_id]

    def migrations(self) -> list[Migration]:
        return [self._migrations[k] for k in sorted(self._migrations)]

    def downgrades_from(self, ref: str) -> list[Migration]:
        return list(self._downgrades.get(ref, ()))

    def validate_migration(self, migration: Migration) -> list[str]:
        """Static checks — what ``llmbic migration validate`` reports."""

        problems: list[str] = []
        for ref in (migration.from_schema, migration.to_schema):
            if not self.has_schema(ref):
                problems.append(f"schema {ref!r} is not registered")
        if migration.from_schema == migration.to_schema:
            problems.append("a migration must change the schema version")

        if problems:
            return problems

        source = self.schema(migration.from_schema)
        target = self.schema(migration.to_schema)

        try:
            migration.ordered_steps()
        except RegistryError as exc:
            problems.append(exc.message)

        for step in migration.steps:
            for fid in step.writes:
                if target.get(fid) is None:
                    problems.append(
                        f"step {step.id!r} writes {fid!r}, which {target.ref} does not define"
                    )
            for ref in step.reads:
                if ref.kind == "field" and source.get(ref.name) is None and target.get(ref.name) is None:
                    problems.append(
                        f"step {step.id!r} reads field {ref.name!r}, which neither "
                        f"{source.ref} nor {target.ref} defines"
                    )
                if ref.kind == "recipe" and ref.name not in self._recipes:
                    problems.append(f"step {step.id!r} reads unknown recipe {ref.name!r}")
                if ref.kind == "vocab" and ref.name not in self._vocabularies:
                    problems.append(f"step {step.id!r} reads unknown vocabulary {ref.name!r}")
            if step.recipe and step.recipe not in self._recipes:
                problems.append(f"step {step.id!r} names unknown recipe {step.recipe!r}")

        problems.extend(self._unhandled_changes(migration, source, target))
        return problems

    def _unhandled_changes(
        self, migration: Migration, source: NormalizedSchema, target: NormalizedSchema
    ) -> list[str]:
        """Every changed field needs an explicit disposition (§21.2).

        This is the check that stops a schema change from quietly shipping with
        half its fields undeclared.
        """

        problems: list[str] = []
        written = set(migration.writes)
        diff = diff_schemas(source, target, renames=migration.renames)

        for change in diff.changes:
            if change.kind.value in ("description_changed", "recipe_changed", "constraint_widened"):
                # Semantic-only changes are handled by recipe currency, not by
                # a step; the planner will notice them.
                continue
            if change.kind.value in ("collection_added", "collection_removed"):
                continue
            if change.kind.value == "removed":
                continue
            if change.kind.value == "renamed" and change.detail.get("identity_preserved"):
                # Same logical field, new path: nothing to move.
                continue
            if change.field_id in written:
                continue
            if change.field_id in migration.acknowledged:
                continue
            problems.append(
                f"{change.kind.value} on {change.to_path or change.from_path!r} "
                f"(field {change.field_id!r}) has no explicit disposition: add a step "
                f"that writes it, or acknowledge it with a rationale"
            )
        return problems

    def _assert_no_cycle(self, migration: Migration) -> None:
        """FR-MIG-004: reject a cycle in the upgrade graph."""

        if migration.to_schema == migration.from_schema:
            raise RegistryError(
                f"migration {migration.id!r} is a self-loop",
                code=ErrorCode.MIGRATION_CYCLE,
            )
        if self._reachable(migration.to_schema, migration.from_schema):
            raise RegistryError(
                f"migration {migration.id!r} ({migration.from_schema} -> {migration.to_schema}) "
                f"would create a cycle; mark it is_downgrade=True to store it as a "
                "reversible downgrade path instead",
                code=ErrorCode.MIGRATION_CYCLE,
                details={"from": migration.from_schema, "to": migration.to_schema},
            )

    def _reachable(self, start: str, target: str) -> bool:
        stack = [start]
        seen: set[str] = set()
        while stack:
            node = stack.pop()
            if node == target:
                return True
            if node in seen:
                continue
            seen.add(node)
            stack.extend(m.to_schema for m in self._upgrades.get(node, ()))
        return False

    def _assert_unique_approved(self, migration: Migration) -> None:
        """FR-MIG-012: one approved production edge per source/target pair."""

        if not migration.approved or migration.branch != "main":
            return
        for other in self._upgrades.get(migration.from_schema, ()):
            if (
                other.to_schema == migration.to_schema
                and other.approved
                and other.branch == "main"
            ):
                raise RegistryError(
                    f"{migration.from_schema} -> {migration.to_schema} already has an "
                    f"approved production migration ({other.id!r}); register "
                    f"{migration.id!r} on a branch or mark it unapproved",
                    code=ErrorCode.AMBIGUOUS_MIGRATION_PATH,
                    details={"existing": other.id},
                )

    # ---- path finding ----------------------------------------------------
    def find_path(
        self,
        from_ref: str,
        to_ref: str,
        *,
        preference: PathPreference | None = None,
    ) -> list[Migration]:
        """The single approved path, chosen deterministically."""

        paths = self.find_paths(from_ref, to_ref, preference=preference)
        if not paths:
            raise PathError(
                f"no migration path from {from_ref} to {to_ref}",
                code=ErrorCode.NO_MIGRATION_PATH,
                details={"from": from_ref, "to": to_ref},
            )
        pref = preference or PathPreference()
        return min(paths, key=pref.key)

    def find_paths(
        self,
        from_ref: str,
        to_ref: str,
        *,
        preference: PathPreference | None = None,
    ) -> list[list[Migration]]:
        """Every valid path, for the planner to compare (FR-PLN-007)."""

        pref = preference or PathPreference()
        if from_ref == to_ref:
            return [[]]
        if not self.has_schema(from_ref):
            raise RegistryError(f"unknown schema {from_ref!r}", code=ErrorCode.SCHEMA_NOT_FOUND)
        if not self.has_schema(to_ref):
            raise RegistryError(f"unknown schema {to_ref!r}", code=ErrorCode.SCHEMA_NOT_FOUND)

        results: list[list[Migration]] = []

        def walk(node: str, acc: list[Migration], visited: frozenset[str]) -> None:
            if len(acc) > pref.max_hops:
                return
            for m in sorted(self._upgrades.get(node, ()), key=lambda m: m.id):
                if pref.require_approved and not m.approved:
                    continue
                if m.branch not in pref.branches:
                    continue
                if pref.forbid_full_document and _permits_full_document(m):
                    continue
                if m.to_schema in visited:
                    continue
                nxt = acc + [m]
                if m.to_schema == to_ref:
                    results.append(nxt)
                    continue
                walk(m.to_schema, nxt, visited | {m.to_schema})

        walk(from_ref, [], frozenset({from_ref}))
        results.sort(key=lambda p: tuple(m.id for m in p))
        return results

    def reachable_versions(self, from_ref: str) -> list[str]:
        seen: set[str] = set()
        stack = [from_ref]
        while stack:
            node = stack.pop()
            for m in self._upgrades.get(node, ()):
                if m.to_schema not in seen:
                    seen.add(m.to_schema)
                    stack.append(m.to_schema)
        return sorted(seen)

    # ---- serialisation ---------------------------------------------------
    def registry_hash(self) -> str:
        """Identity of the whole registry state — part of a plan's signature."""

        return content_hash(
            {
                "schemas": {k: v.schema_hash() for k, v in sorted(self._schemas.items())},
                "migrations": {
                    k: v.migration_hash() for k, v in sorted(self._migrations.items())
                },
                "recipes": {k: v.recipe_hash() for k, v in sorted(self._recipes.items())},
                "vocabularies": {
                    k: v.vocabulary_hash() for k, v in sorted(self._vocabularies.items())
                },
            }
        )

    def to_canonical(self) -> dict[str, Any]:
        return {
            "schemas": [s.to_canonical() for s in self.schemas()],
            "migrations": [m.to_canonical() for m in self.migrations()],
            "recipes": [r.to_canonical() for r in self.recipes()],
            "vocabularies": [v.to_canonical() for v in self.vocabularies()],
        }

    @classmethod
    def from_canonical(cls, data: Mapping[str, Any]) -> "Registry":
        reg = cls()
        for s in data.get("schemas") or ():
            reg.register_schema(NormalizedSchema.from_canonical(s))
        for v in data.get("vocabularies") or ():
            reg.register_vocabulary(Vocabulary.from_canonical(v))
        for r in data.get("recipes") or ():
            reg.register_recipe(ExtractionRecipe.from_canonical(r))
        for m in data.get("migrations") or ():
            reg.register_migration(Migration.from_canonical(m), validate=False)
        return reg


def _permits_full_document(migration: Migration) -> bool:
    for step in migration.steps:
        if step.context is not None and step.context.permits_full_document():
            return True
    return False


__all__ = ["PathPreference", "Registry", "SchemaFamily"]
