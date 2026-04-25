FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

# Copy the root mempalace package (pyproject.toml + mempalace/ dir)
COPY pyproject.toml ./mempalace-core/pyproject.toml
COPY mempalace/ ./mempalace-core/mempalace/

# Copy pgvector backend package
COPY mempalace-pgvector/pyproject.toml ./mempalace-pgvector/pyproject.toml
COPY mempalace-pgvector/src/ ./mempalace-pgvector/src/

# Copy server package
COPY mempalace-server/pyproject.toml ./mempalace-server/pyproject.toml
COPY mempalace-server/src/ ./mempalace-server/src/

RUN uv pip install --system \
    ./mempalace-core \
    ./mempalace-pgvector \
    ./mempalace-server

EXPOSE 15033

CMD ["uvicorn", "mempalace_server.app:app", "--host", "0.0.0.0", "--port", "15033"]
