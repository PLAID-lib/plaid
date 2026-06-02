#!/usr/bin/env bash
set -euo pipefail

# This script is intentionally the single documentation command used locally and
# in CI. It prepares Jupytext notebooks, executes them so outputs are rendered in
# the generated documentation, and then delegates the site build to Zensical.

DOCS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${DOCS_DIR}/.." && pwd)"

cd "${DOCS_DIR}"

echo "#---# Clean generated documentation artifacts"
rm -rf _build site .cache source/notebooks

echo "#---# Regenerate mkdocstrings API reference stubs"
python "${DOCS_DIR}/generate_api_stubs.py"

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

echo "#---# Remove notebook source files that should not be published as static assets"
find source/notebooks -name "*.py" -delete
find source/notebooks -name "*.ipynb" -delete
find source/notebooks -name "*.bat" -delete
find source/notebooks -name "*.sh" -delete

echo "#---# Build the static site with Zensical"
zensical build --clean --config-file zensical.toml

echo "#---# Copy custom stylesheets"
mkdir -p _build/html/stylesheets
cp -av stylesheets/*.css _build/html/stylesheets/