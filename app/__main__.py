"""``python -m app`` starts the API using SJ_HOST / SJ_PORT from the environment or .env."""
import uvicorn

from app.core.config import get_settings

if __name__ == "__main__":
    s = get_settings()
    uvicorn.run("app.main:create_app", factory=True, host=s.host, port=s.port,
                log_level=s.log_level.lower())
