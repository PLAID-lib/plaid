# run_examples.ps1

param(
  [string]$pythonPath = "python"
)

$files = Get-ChildItem -Path . -Filter *.py
$utils = Get-ChildItem -Path utils -Filter *.py
$containers = Get-ChildItem -Path containers -Filter *.py
$post = Get-ChildItem -Path post -Filter *.py

$allFiles = $files + $utils + $containers + $post

foreach ($file in $allFiles) {
    Write-Host "--------------------------------------------------------------------------------------"
    Write-Host "#---# run python $($file.FullName)"
    & $pythonPath $file.FullName
    if ($LASTEXITCODE -ne 0) {
        exit 1
    }
}