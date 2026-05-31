"""Generate the mkdocstrings API reference for the ``plaid`` package.

This helper walks ``src/plaid`` and creates a minimal Markdown stub for each
module and (sub)package under ``docs/source/api``. Each stub contains a single
``:::`` mkdocstrings directive; rendering options are centralised in
``docs/zensical.toml`` (``[plugins.mkdocstrings.handlers.python]``).

It also rewrites the ``API reference`` ``nav`` block in ``docs/zensical.toml``
between the markers::

    # >>> AUTO-GENERATED API REFERENCE START
    ...
    # <<< AUTO-GENERATED API REFERENCE END

so the navigation tree always mirrors the source layout.

Usage:

    python docs/generate_api_stubs.py

The script is idempotent and is intended to be run whenever modules are added,
removed, or renamed under ``src/plaid``. CI runs it and fails if the working
tree changes (stubs or ``zensical.toml`` out of sync with ``src/plaid``).
"""

from __future__ import annotations

from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent
REPO_DIR = DOCS_DIR.parent
SRC_DIR = REPO_DIR / "src" / "plaid"
API_DIR = DOCS_DIR / "source" / "api"
ZENSICAL_CONFIG = DOCS_DIR / "zensical.toml"

NAV_START = "# >>> AUTO-GENERATED API REFERENCE START"
NAV_END = "# <<< AUTO-GENERATED API REFERENCE END"

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


def render_nav_block(base_indent: str) -> str:
    """Render the TOML ``API reference`` nav block mirroring the source tree.

    ``base_indent`` is the leading whitespace of the surrounding nav entries in
    ``zensical.toml`` so that the generated block fits the existing style.
    """
    documented_dirs, files = collect_modules()

    def render(directory: Path, depth: int) -> list[str]:
        rel = directory.relative_to(SRC_DIR)
        indent = base_indent + "  " * depth
        lines: list[str] = []

        index_rel = "/".join(("api", *rel.parts, "index.md")) if rel.parts else "api/index.md"
        lines.append(f'{indent}{{ "Overview" = "{index_rel}" }},')

        module_files = sorted(
            f for f in files
            if f.parent == API_DIR.joinpath(*rel.parts) and f.name != "index.md"
        )
        for f in module_files:
            rel_path = "/".join(("api", *rel.parts, f.name)) if rel.parts else f"api/{f.name}"
            lines.append(f'{indent}{{ "{f.stem}" = "{rel_path}" }},')

        sub_dirs = sorted(
            d for d in documented_dirs
            if d.parent == directory and d != directory
        )
        for sub in sub_dirs:
            lines.append(f'{indent}{{ "{sub.name}" = [')
            lines.extend(render(sub, depth + 1))
            lines.append(f"{indent}] }},")

        return lines

    inner = render(SRC_DIR, depth=1)
    return "\n".join(
        [f'{base_indent}{{ "API reference" = [', *inner, f"{base_indent}] }},"]
    )


def update_zensical_nav() -> bool:
    """Rewrite the API-reference block in ``zensical.toml`` between markers.

    Returns ``True`` when the file was modified.
    """
    if not ZENSICAL_CONFIG.is_file():
        raise SystemExit(f"Zensical config not found: {ZENSICAL_CONFIG}")

    text = ZENSICAL_CONFIG.read_text(encoding="utf-8")
    lines = text.splitlines()

    start_idx = end_idx = None
    start_indent = ""
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith(NAV_START):
            start_idx = i
            start_indent = line[: len(line) - len(stripped)]
        elif stripped.startswith(NAV_END):
            end_idx = i
            break

    if start_idx is None or end_idx is None or end_idx <= start_idx:
        raise SystemExit(
            f"Could not locate AUTO-GENERATED markers in {ZENSICAL_CONFIG.relative_to(REPO_DIR)}.\n"
            f"Expected lines starting with:\n  {NAV_START}\n  {NAV_END}"
        )

    header = (
        f"{start_indent}{NAV_START}\n"
        f"{start_indent}# The block below is overwritten by `python docs/generate_api_stubs.py`.\n"
        f"{start_indent}# Edit that script (or the markers) instead of changing this section by hand."
    )
    block = render_nav_block(start_indent)
    footer = f"{start_indent}{NAV_END}"

    new_section = "\n".join([header, block, footer])
    new_lines = lines[:start_idx] + new_section.splitlines() + lines[end_idx + 1 :]
    new_text = "\n".join(new_lines)
    if text.endswith("\n") and not new_text.endswith("\n"):
        new_text += "\n"

    if new_text != text:
        ZENSICAL_CONFIG.write_text(new_text, encoding="utf-8")
        return True
    return False


def main() -> None:
    if not SRC_DIR.is_dir():
        raise SystemExit(f"Source directory not found: {SRC_DIR}")

    written = write_stubs()
    removed = prune_stale(written)
    nav_changed = update_zensical_nav()

    print(f"Wrote {len(written)} stub(s) under {API_DIR.relative_to(REPO_DIR)}")
    if removed:
        print(f"Removed {len(removed)} stale file(s):")
        for p in removed:
            print(f"  - {p.relative_to(REPO_DIR)}")
    if nav_changed:
        print(f"Updated API-reference nav block in {ZENSICAL_CONFIG.relative_to(REPO_DIR)}")
    else:
        print(f"API-reference nav block in {ZENSICAL_CONFIG.relative_to(REPO_DIR)} already up to date")


if __name__ == "__main__":
    main()
