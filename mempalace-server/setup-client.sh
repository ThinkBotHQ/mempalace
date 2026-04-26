#!/bin/bash
# MemPalace Remote Setup — run this once to connect to the shared memory server
#
# Usage:
#   curl -s https://raw.githubusercontent.com/ThinkBotHQ/mempalace/develop/mempalace-server/setup-client.sh | bash -s -- <your-api-key>
#
# Or locally:
#   bash setup-client.sh <your-api-key>

set -e

API_KEY="$1"

if [ -z "$API_KEY" ]; then
    echo "Usage: bash setup-client.sh <your-api-key>"
    echo ""
    echo "Get your key from your team lead."
    exit 1
fi

if ! echo "$API_KEY" | grep -q "^mp_live_"; then
    echo "Error: API key must start with 'mp_live_'"
    exit 1
fi

echo "Setting up MemPalace remote connection..."

# 1. Add the remote MCP server
claude mcp remove mempalace 2>/dev/null || true
claude mcp add --transport http --scope user mempalace https://mp.thinks.bot/mcp \
    --header "Authorization: Bearer $API_KEY"

echo "  ✓ Remote MCP server added (mp.thinks.bot)"

# 2. Disable any local MCP server from the plugin (if installed)
PLUGIN_DIR="$HOME/.claude/plugins/marketplaces/mempalace/.claude-plugin"
CACHE_DIR="$HOME/.claude/plugins/cache/mempalace"

for f in \
    "$PLUGIN_DIR/plugin.json" \
    "$PLUGIN_DIR/.mcp.json" \
    "$CACHE_DIR/mempalace/3.3.2/plugin.json" \
    "$CACHE_DIR/mempalace/3.3.2/.mcp.json"; do
    if [ -f "$f" ]; then
        python3 -c "
import json
with open('$f') as fh:
    d = json.load(fh)
changed = False
if 'mcpServers' in d and d['mcpServers']:
    d['mcpServers'] = {}
    changed = True
if isinstance(d, dict) and 'mempalace' in d and 'command' in d.get('mempalace', {}):
    d = {}
    changed = True
if changed:
    with open('$f', 'w') as fh:
        json.dump(d, fh, indent=2)
" 2>/dev/null && echo "  ✓ Disabled local MCP in $(basename $f)"
    fi
done

# 3. Disable local mining hooks from the plugin
HOOKS_FILE="$PLUGIN_DIR/hooks/hooks.json"
if [ -f "$HOOKS_FILE" ]; then
    python3 -c "
import json
with open('$HOOKS_FILE') as f:
    d = json.load(f)
if d.get('hooks'):
    d['hooks'] = {}
    with open('$HOOKS_FILE', 'w') as f:
        json.dump(d, f, indent=2)
" 2>/dev/null && echo "  ✓ Disabled plugin hooks (using remote hooks)"
fi

echo ""
echo "Done! Restart Claude Code to connect."
echo ""
echo "Verify with: claude mcp list | grep mempalace"
echo "Should show: mempalace: https://mp.thinks.bot/mcp (HTTP) - ✓ Connected"
