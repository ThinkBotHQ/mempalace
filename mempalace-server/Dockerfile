FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

# Copy the root mempalace package (pyproject.toml + mempalace/ dir)
COPY pyproject.toml ./mempalace-core/pyproject.toml
COPY README.md ./mempalace-core/README.md
COPY mempalace/ ./mempalace-core/mempalace/

# Copy pgvector backend package
COPY mempalace-pgvector/pyproject.toml ./mempalace-pgvector/pyproject.toml
COPY mempalace-pgvector/src/ ./mempalace-pgvector/src/

# Copy server package
COPY mempalace-server/pyproject.toml ./mempalace-server/pyproject.toml
COPY mempalace-server/src/ ./mempalace-server/src/

# Install each package explicitly from local paths.
# --no-deps on mempalace-server avoids PyPI resolution for local siblings.
RUN uv pip install --system ./mempalace-core ./mempalace-pgvector \
    && uv pip install --system --no-deps ./mempalace-server \
    && uv pip install --system "mcp>=1.0,<3.0" "starlette>=1.0,<2.0" "uvicorn[standard]>=0.30,<1.0" "httpx>=0.27,<1.0" "psycopg[binary]>=3.2.0,<4.0"

EXPOSE 15033

CMD ["mempalace-server"]
