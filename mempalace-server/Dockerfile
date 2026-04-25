FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY mempalace/ mempalace/
COPY mempalace-pgvector/ mempalace-pgvector/
COPY mempalace-server/ mempalace-server/
COPY pyproject.toml .

RUN uv pip install --system \
    ./mempalace \
    ./mempalace-pgvector \
    ./mempalace-server

EXPOSE 15033

CMD ["uvicorn", "mempalace_server.app:app", "--host", "0.0.0.0", "--port", "15033"]
