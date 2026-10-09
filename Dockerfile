# Speech Judge image. One image, two roles (api / worker) selected by the command.
#   GPU server:  docker build -t speech-judge .
#   CPU / mock:  docker build --build-arg INSTALL_ML=false -t speech-judge:mock .
FROM python:3.12-slim

ARG INSTALL_ML=true
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 \
    SJ_HOST=0.0.0.0 SJ_PORT=8000

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv
COPY requirements.txt requirements-ml.txt ./
RUN pip install -r requirements.txt \
    && if [ "$INSTALL_ML" = "true" ]; then pip install -r requirements-ml.txt; fi

COPY app ./app
COPY config ./config

RUN useradd --create-home --uid 10001 judge
USER judge

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=5 \
    CMD python -c "import os,urllib.request as u; u.urlopen('http://127.0.0.1:%s/api/v1/health' % os.environ.get('SJ_PORT','8000'), timeout=2)"

# Default role: API. The worker overrides the command in docker-compose.yml.
CMD ["python", "-m", "app"]
