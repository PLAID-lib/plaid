#!/bin/bash

#---# Clean
rm -rf _build
rm -rf source/additional
rm -rf autoapi
# rm -rf api_reference
# mkdir -p source/additional
# mkdir -p _extra

#---# Copy coverage in local
# rm -rf _extra/coverage
# cp -r ../public/coverage _extra/coverage

#---# Copy diagrams in local
# rm -rf _extra/diagrams
# cp -r ../public/diagrams _extra/diagrams
# rm -rf source/additional/diagrams
# cp -r ../public/diagrams source/additional/diagrams

# #---# Copy notebooks in local
# rm -rf source/additional/notebooks
# mkdir -p source/additional/notebooks
# cp ../examples/*.ipynb source/additional/notebooks

# #---# Generate API pages
# sphinx-apidoc --ext-autodoc --ext-doctest --ext-intersphinx --ext-todo --ext-coverage --ext-imgmath --ext-mathjax --ext-ifconfig --ext-viewcode --ext-githubpages -P -e -f -d 2 -o api_reference/plaid ../src/plaid
# sphinx-apidoc --ext-autodoc --ext-doctest --ext-intersphinx --ext-todo --ext-coverage --ext-imgmath --ext-mathjax --ext-ifconfig --ext-viewcode --ext-githubpages -P -e -f -d 2 -o api_reference/tests ../tests
# sphinx-apidoc --ext-autodoc --ext-doctest --ext-intersphinx --ext-todo --ext-coverage --ext-imgmath --ext-mathjax --ext-ifconfig --ext-viewcode --ext-githubpages -P -e -f -d 2 -o api_reference/examples ../examples
# #---# Clean and add informations to generated API pages
# python fix_module.py

#---# Verbose BEFORE
echo "";echo "#---# ls -lAh source/"
ls -lAh source/
# echo "";echo "#---# ls -lAh source/additional/*"
# ls -lAh source/additional/notebooks
# echo "";echo "#---# ls -lAh _extra/diagrams/*"
# ls -lAh _extra/diagrams/*
# echo "";echo "#---# ls -lAh api_reference/"
# ls -lAh api_reference/
# cat api_reference/*/module*

#---# Generate doc site
# export PYDEVD_DISABLE_FILE_VALIDATION=1
make html

#---# Verbose AFTER
echo "";echo "#---# ls -lAh"
ls -lAh *
echo "";echo "#---# ls -lAh _build/html/*"
ls -lAh _build/html/*
echo "";echo "#---# ls -lAh _build/html/autoapi"
ls -lAh _build/html/autoapi
echo "";echo "#---# ls -lAh _build/html/reports"
ls -lAh _build/html/reports

#---# Post
rsync -av --delete-after _build/html/* ../public
