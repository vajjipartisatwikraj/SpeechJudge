# Start the API using speech-judge/.env (host/port/credentials/models all come from there).
#   .\scripts\run_api.ps1            real models (needs Ollama running with the configured model)
#   .\scripts\run_api.ps1 -Mock      everything mocked: instant start, no model downloads
param([switch]$Mock)

Set-Location (Join-Path $PSScriptRoot "..")
if (-not (Test-Path ".env")) {
    Write-Error ".env not found. Copy .env.example to .env and edit it."; exit 1
}
if ($Mock) { $env:SJ_MOCK_MODE = "true"; Write-Host "Mock mode: no real models are used." -ForegroundColor Yellow }
& .\.venv\Scripts\python.exe -m app
