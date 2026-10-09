# Start a Celery worker (needs Redis and SJ_JOB_BACKEND=celery in .env).
# On Windows the solo pool is used; on Linux servers use --pool=threads (see docker-compose.yml).
Set-Location (Join-Path $PSScriptRoot "..")
& .\.venv\Scripts\celery.exe -A app.workers.celery_app worker --pool=solo --loglevel=INFO
