"""Generate the mkdocstrings API reference stubs for the ``plaid`` package.

This helper walks ``src/plaid`` and creates a minimal Markdown stub for each
module and (sub)package under ``docs/source/api``. Each stub contains a single
``:::`` mkdocstrings directive; rendering options are centralised in
``docs/zensical.toml`` (``[plugins.mkdocstrings.handlers.python]``).

It also prints a suggested ``nav`` block to paste into ``docs/zensical.toml`` so
that the curated navigation can be kept in sync with the source layout.

Usage:

    python docs/generate_api_stubs.py

The script is idempotent and is intended to be run whenever modules are added,
removed, or renamed under ``src/plaid``. CI runs it and fails if the resulting
files differ from what is committed.
"""

from __future__ import annotations

from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent
REPO_DIR = DOCS_DIR.parent
SRC_DIR = REPO_DIR / "src" / "plaid"
API_DIR = DOCS_DIR / "source" / "api"

SKIP_FILES = {"_version.py"}


def module_name(path: Path) -> str:
    rel = path.relative_to(SRC_DIR).with_suffix("")
    if rel.name == "__init__":
        rel = rel.parent
    return ".".join(("plaid", *rel.parts)) if rel.parts else "plaid"


def output_path(path: Path) -> Path:
    rel = path.relative_to(SRC_DIR)
    if rel.name == "__init__.py":
        return API_DIR.joinpath(*rel.parent.parts, "index.md")
    return API_DIR.joinpath(*rel.with_suffix(".md").parts)


def is_namespace_package(path: Path) -> bool:
    return path.name == "__init__.py" and not path.read_text(encoding="utf-8").strip()


def stub_content(module: str) -> str:
    return f"# `{module}`\n\n::: {module}\n"


def collect_modules() -> tuple[set[Path], dict[Path, str]]:
    """Return (documented directories, mapping output path -> module name)."""
    documented_dirs: set[Path] = {SRC_DIR}
    files: dict[Path, str] = {}

    for path in sorted(SRC_DIR.rglob("*.py")):
        if path.name in SKIP_FILES or "__pycache__" in path.parts:
            continue

        # Track every parent directory for the package index pages.
        current = path.parent
        while True:
            documented_dirs.add(current)
            if current == SRC_DIR:
                break
            current = current.parent

        # Skip empty namespace ``__init__`` files: the package index is created
        # below from ``documented_dirs``.
        if path.name == "__init__.py" and is_namespace_package(path):
            continue

        files[output_path(path)] = module_name(path)

    return documented_dirs, files


def write_stubs() -> list[Path]:
    documented_dirs, files = collect_modules()

    # Module / non-empty ``__init__`` pages.
    written: list[Path] = []
    for out, module in files.items():
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(stub_content(module), encoding="utf-8")
        written.append(out)

    # Package index pages for namespace packages (empty ``__init__``).
    for directory in documented_dirs:
        rel = directory.relative_to(SRC_DIR)
        out = API_DIR.joinpath(*rel.parts, "index.md") if rel.parts else API_DIR / "index.md"
        if out in written:
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        module = ".".join(("plaid", *rel.parts)) if rel.parts else "plaid"
        out.write_text(stub_content(module), encoding="utf-8")
        written.append(out)

    return written


def prune_stale(written: list[Path]) -> list[Path]:
    """Delete any ``.md`` file under ``API_DIR`` not produced by this run."""
    kept = {p.resolve() for p in written}
    removed: list[Path] = []
    if not API_DIR.exists():
        return removed
    for md in API_DIR.rglob("*.md"):
        if md.resolve() not in kept:
            md.unlink()
            removed.append(md)
    # Drop any now-empty directories.
    for directory in sorted(
        (p for p in API_DIR.rglob("*") if p.is_dir()),
        key=lambda p: len(p.parts),
        reverse=True,
    ):
        if not any(directory.iterdir()):
            directory.rmdir()
    return removed


def suggested_nav() -> str:
    """Return a TOML ``nav`` snippet mirroring the generated stubs.

    The output is meant to be copy/pasted into the ``API reference`` entry in
    ``docs/zensical.toml`` whenever the source layout changes.
    """
    documented_dirs, files = collect_modules()

    def render(directory: Path, indent: int) -> list[str]:
        rel = directory.relative_to(SRC_DIR)
        prefix = "    " * indent
        lines: list[str] = []
        # ``Overview`` entry for the package itself.
        index_rel = "/".join(("api", *rel.parts, "index.md")) if rel.parts else "api/index.md"
        lines.append(f'{prefix}{{ "Overview" = "{index_rel}" }},')

        # Module pages directly inside this package.
        module_files = sorted(
            f for f in files
            if f.parent == API_DIR.joinpath(*rel.parts) and f.name != "index.md"
        )
        for f in module_files:
            stem = f.stem
            rel_path = "/".join(("api", *rel.parts, f.name)) if rel.parts else f"api/{f.name}"
            lines.append(f'{prefix}{{ "{stem}" = "{rel_path}" }},')

        # Sub-packages.
        sub_dirs = sorted(
            d for d in documented_dirs
            if d.parent == directory and d != directory
        )
        for sub in sub_dirs:
            name = sub.name
            lines.append(f'{prefix}{{ "{name}" = [')
            lines.extend(render(sub, indent + 1))
            lines.append(f"{prefix}] }},")

        return lines

    body = render(SRC_DIR, indent=2)
    return "\n".join(
        ['  { "API reference" = [', *body, "  ] },"]
    )


def main() -> None:
    if not SRC_DIR.is_dir():
        raise SystemExit(f"Source directory not found: {SRC_DIR}")

    written = write_stubs()
    removed = prune_stale(written)

    print(f"Wrote {len(written)} stub(s) under {API_DIR.relative_to(REPO_DIR)}")
    if removed:
        print(f"Removed {len(removed)} stale file(s):")
        for p in removed:
            print(f"  - {p.relative_to(REPO_DIR)}")

    print()
    print("Suggested zensical.toml nav block (copy/paste under [project] nav):")
    print()
    print(suggested_nav())


if __name__ == "__main__":
    main()
