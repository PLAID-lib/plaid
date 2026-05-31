@echo off
for %%f in (*.py containers\*.py) do (
  echo --------------------------------------------------------------------------------------
  echo #---# run python %%f
  python %%f || exit /b 1
)