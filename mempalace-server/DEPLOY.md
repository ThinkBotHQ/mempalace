# MemPalace Remote MCP Server — Deployment Guide

## What this is

A remote MCP server at `https://mp.thinks.bot/mcp` that lets any Claude Code
session connect to shared MemPalace memories with one command.

## Prerequisites on server

- Docker + Docker Compose
- Nginx
- Certbot (for Let's Encrypt TLS)
- Git

## Step-by-step deployment

### 1. Clone the repo

```bash
git clone https://github.com/ThinkBotHQ/mempalace.git
cd mempalace
```

### 2. Create .env file

```bash
cat > mempalace-server/.env << 'EOF'
POSTGRES_PASSWORD=<generate-a-strong-password>
GEMINI_API_KEY=<your-gemini-api-key>
EOF
chmod 600 mempalace-server/.env
```

### 3. Start services

```bash
cd mempalace-server
docker compose --env-file .env up -d
```

This starts:
- **pgvector** (PostgreSQL 17 + pgvector) on internal port 5432
- **mcp** (mempalace-server) on port 15033

### 4. Set up schema

First time only — initialize the pgvector database with MemPalace tables:

```bash
# Get the pgvector container name
docker compose exec pgvector psql -U postgres mempalace < ../mempalace-pgvector/src/mempalace_pgvector/schema.sql
```

### 5. Migrate data from local machine

On your local machine (where your current 343K docs live):

```bash
# Dump
docker exec mempalace-pgvector pg_dump -U postgres mempalace | gzip > /tmp/mempalace_dump.sql.gz

# Transfer to server
scp /tmp/mempalace_dump.sql.gz your-server:/tmp/

# On server — restore
gunzip -c /tmp/mempalace_dump.sql.gz | docker compose exec -T pgvector psql -U postgres mempalace
```

### 6. Create API keys

```bash
# Enter the MCP container
docker compose exec mcp mempalace-api-key create justin-local
docker compose exec mcp mempalace-api-key create team-member-1

# List all keys
docker compose exec mcp mempalace-api-key list

# Revoke a key
docker compose exec mcp mempalace-api-key revoke <name>
```

Save the printed tokens — they won't be shown again.

### 7. Set up Nginx + TLS

```bash
# Install certbot if needed
sudo apt install certbot python3-certbot-nginx

# Get certificate
sudo certbot certonly --nginx -d mp.thinks.bot

# Copy nginx config
sudo cp mempalace-server/nginx.conf /etc/nginx/sites-available/mp.thinks.bot
sudo ln -sf /etc/nginx/sites-available/mp.thinks.bot /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

### 8. Verify

```bash
# Health check (no auth)
curl https://mp.thinks.bot/health

# MCP initialize (with auth)
curl -s https://mp.thinks.bot/mcp/ -X POST \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -H "Authorization: Bearer mp_live_<your-key>" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}'
```

## Team member setup

Each team member runs ONE command:

```bash
claude mcp add --transport http --scope user mempalace https://mp.thinks.bot/mcp \
  --header "Authorization: Bearer mp_live_<their-key>"
```

Or add to `.claude/settings.json`:

```json
{
  "mcpServers": {
    "mempalace": {
      "type": "http",
      "url": "https://mp.thinks.bot/mcp",
      "headers": {
        "Authorization": "Bearer ${MEMPALACE_API_KEY}"
      }
    }
  }
}
```

Then set `MEMPALACE_API_KEY=mp_live_<their-key>` in their shell.

All 35 tools appear in their Claude Code sessions immediately.

## Monitoring

```bash
# Server logs
docker compose logs -f mcp

# Database size
docker compose exec pgvector psql -U postgres mempalace -c \
  "SELECT pg_size_pretty(pg_database_size('mempalace'));"

# Document count
docker compose exec pgvector psql -U postgres mempalace -c \
  "SELECT COUNT(*) FROM mp_documents;"

# API key usage (check rate limit hits in logs)
docker compose logs mcp | grep "rate_limit"
```

## Backup

```bash
# Daily pg_dump cron
echo "0 3 * * * docker compose -f /path/to/mempalace-server/docker-compose.yml exec -T pgvector pg_dump -U postgres mempalace | gzip > /backups/mempalace_\$(date +\%Y\%m\%d).sql.gz" | crontab -
```

## Troubleshooting

**Server won't start:**
- Check `MEMPALACE_PGVECTOR_DSN` in docker-compose.yml matches the password in `.env`
- Check `docker compose logs mcp` for Python errors

**Auth failures:**
- Verify key exists: `docker compose exec mcp mempalace-api-key list`
- Verify key is active (not revoked)
- Check the Bearer token format: `Authorization: Bearer mp_live_...`

**Slow first request:**
- The first request after startup warms the embedding model + DB connections
- Subsequent requests should be <100ms for reads, 1-3s for writes (embedding)

**502 Bad Gateway from Nginx:**
- Check MCP container is running: `docker compose ps`
- Check port 15033 is exposed: `curl http://localhost:15033/health`
