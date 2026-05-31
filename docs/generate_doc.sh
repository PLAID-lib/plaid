#!/usr/bin/env bash
set -euo pipefail

# This script is intentionally the single documentation command used locally and
# in CI. It prepares Jupytext notebooks, executes them so outputs are rendered in
# the generated documentation, and then delegates the site build to Zensical.

DOCS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${DOCS_DIR}/.." && pwd)"

cd "${DOCS_DIR}"

echo "#---# Clean generated documentation artifacts"
rm -rf _build site .cache source/notebooks source/api

echo "#---# Copy Jupytext notebooks from examples/"
mkdir -p source/notebooks
cp -R "${REPO_DIR}/examples/." source/notebooks/

echo "#---# Convert Jupytext Python notebooks to ipynb"
find source/notebooks -name "*_example.py" -print0 \
  | xargs -0 -r -n 1 jupytext --to ipynb

echo "#---# Execute notebooks and export Markdown with outputs"
while IFS= read -r -d '' notebook; do
  notebook_dir="$(dirname "${notebook}")"
  notebook_base="$(basename "${notebook}" .ipynb)"

  jupyter nbconvert \
    --to markdown \
    --execute \
    --ExecutePreprocessor.timeout=300 \
    --output "${notebook_base}" \
    --output-dir "${notebook_dir}" \
    "${notebook}"

  # Remove noisy HTTP client logs from notebook outputs. They render poorly as
  # Markdown reference-style links while adding no useful information to the
  # published examples.
  sed -i '/:INFO:_client\.py:_send_single_request(1025)\]:HTTP Request:/d' \
    "${notebook_dir}/${notebook_base}.md"
done < <(find source/notebooks -name "*_example.ipynb" -print0)

echo "#---# Generate API reference pages from src/plaid"
python - <<'PY'
from __future__ import annotations

from pathlib import Path


def module_name(path: Path) -> str:
    rel = path.relative_to(src_dir).with_suffix("")
    if rel.name == "__init__":
        rel = rel.parent
    return ".".join(("plaid", *rel.parts)) if rel.parts else "plaid"


def output_path(path: Path) -> Path:
    rel = path.relative_to(src_dir)
    if rel.name == "__init__.py":
        return api_dir.joinpath(*rel.parent.parts, "index.md")
    return api_dir.joinpath(*rel.with_suffix(".md").parts)


def is_namespace_package(path: Path) -> bool:
    return path.name == "__init__.py" and not path.read_text(encoding="utf-8").strip()


src_dir = Path("../src/plaid").resolve()
api_dir = Path("source/api")
api_dir.mkdir(parents=True, exist_ok=True)

documented_dirs: set[Path] = {src_dir}

for path in sorted(src_dir.rglob("*.py")):
    if path.name == "_version.py" or "__pycache__" in path.parts:
        continue

    module = module_name(path)
    out = output_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    current = path.parent
    while current != src_dir.parent:
        documented_dirs.add(current)
        if current == src_dir:
            break
        current = current.parent

    if path.name == "__init__.py" and is_namespace_package(path):
        continue
    if path.name == "__init__.py":
        content = f"# `{module}`\n\n::: {module}\n"
    else:
        content = (
            f"# `{module}`\n\n"
            f"::: {module}\n"
            "    options:\n"
            "      show_source: true\n"
            "      show_root_heading: true\n"
            "      show_root_full_path: false\n"
        )

    out.write_text(content, encoding="utf-8")

for directory in sorted(documented_dirs, key=lambda item: len(item.relative_to(src_dir).parts), reverse=True):
    rel = directory.relative_to(src_dir)
    out = api_dir.joinpath(*rel.parts, "index.md") if rel.parts else api_dir / "index.md"
    if out.exists():
        continue

    module = ".".join(("plaid", *rel.parts)) if rel.parts else "plaid"
    entries = []
    for child in sorted(out.parent.iterdir()):
        if child.name == "index.md":
            continue
        if child.is_dir() and (child / "index.md").exists():
            entries.append(f"- [{child.name}]({child.name}/index.md)")
        elif child.suffix == ".md":
            entries.append(f"- [{child.stem}]({child.name})")

    out.write_text(f"# `{module}`\n\n" + "\n".join(entries) + "\n", encoding="utf-8")
PY

echo "#---# Remove source files that should not be published as static assets"
find source -name "*.rst" -delete
find source/notebooks -name "*.py" -delete
find source/notebooks -name "*.ipynb" -delete
find source/notebooks -name "*.bat" -delete
find source/notebooks -name "*.sh" -delete

echo "#---# Generate Zensical site"
zensical build --clean --config-file zensical.toml

echo "#---# Fix logo links for static file browsing"
python - <<'PY'
from pathlib import Path
import re

site_dir = Path("_build/html")
for html_file in site_dir.rglob("*.html"):
    rel_parent = html_file.parent.relative_to(site_dir)
    if rel_parent == Path("."):
        home = "index.html"
    else:
        home = "/".join([".."] * len(rel_parent.parts) + ["index.html"])

    text = html_file.read_text(encoding="utf-8")
    # Zensical currently generates directory links for the logo even when
    # use_directory_urls=false. Those links open directory listings when the
    # site is browsed from a static file server without directory indexes.
    text = re.sub(
        r'(<a href=")[^"]*(" title="plaid" class="md-header__button md-logo")',
        rf'\g<1>{home}\2',
        text,
        count=1,
    )
    html_file.write_text(text, encoding="utf-8")
PY

if [ -d "${REPO_DIR}/public" ]; then
  echo "#---# Sync generated site to ../public"
  rsync -av --delete-after _build/html/ "${REPO_DIR}/public/"
else
  echo "#---# Skip sync: ../public does not exist"
fi
