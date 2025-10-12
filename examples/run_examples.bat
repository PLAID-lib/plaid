@echo off
for %%f in (*.py examples\*.py utils\*.py containers\*.py post\*.py) do (
  echo --------------------------------------------------------------------------------------
  echo #---# run python %%f
  python %%f || exit /b 1
)