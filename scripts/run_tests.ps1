# Run the test suite.
#   .\scripts\run_tests.ps1                 unit + integration (fast, no models)
#   .\scripts\run_tests.ps1 -Real           also the end-to-end tests (real models; stack tests need the API + frontend running)
#   .\scripts\run_tests.ps1 -Path tests\unit
param([switch]$Real, [string]$Path = "tests")

Set-Location (Join-Path $PSScriptRoot "..")
$pytestArgs = @("-m", "pytest", $Path, "-q")
if ($Real) { $pytestArgs += "--real" }
& .\.venv\Scripts\python.exe @pytestArgs
exit $LASTEXITCODE
