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

HOOKS_DIR="$HOME/.claude/hooks"
PLUGIN_DIR="$HOME/.claude/plugins/marketplaces/mempalace/.claude-plugin"
CACHE_DIR="$HOME/.claude/plugins/cache/mempalace"
SETTINGS="$HOME/.claude/settings.json"
SCRIPT_URL="https://raw.githubusercontent.com/ThinkBotHQ/mempalace/develop"

echo "Setting up MemPalace remote connection..."

# 1. Add the remote MCP server
claude mcp remove mempalace 2>/dev/null || true
claude mcp add --transport http --scope user mempalace https://mp.thinks.bot/mcp \
    --header "Authorization: Bearer $API_KEY"
echo "  ✓ Remote MCP server added (mp.thinks.bot)"

# 2. Disable any local MCP server from the plugin (if installed)
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

# 3. Disable plugin hooks (we install our own that use remote MCP)
for hf in "$PLUGIN_DIR/hooks/hooks.json" "$CACHE_DIR/mempalace/3.3.2/hooks/hooks.json"; do
    if [ -f "$hf" ]; then
        python3 -c "
import json
with open('$hf') as f:
    d = json.load(f)
if d.get('hooks'):
    d['hooks'] = {}
    with open('$hf', 'w') as f:
        json.dump(d, f, indent=2)
" 2>/dev/null
    fi
done
echo "  ✓ Disabled plugin hooks"

# 4. Install save + precompact hooks
mkdir -p "$HOOKS_DIR"

curl -sf "$SCRIPT_URL/hooks/mempal_save_hook.sh" -o "$HOOKS_DIR/mempal_save_hook.sh"
chmod +x "$HOOKS_DIR/mempal_save_hook.sh"

curl -sf "$SCRIPT_URL/hooks/mempal_precompact_hook.sh" -o "$HOOKS_DIR/mempal_precompact_hook.sh"
chmod +x "$HOOKS_DIR/mempal_precompact_hook.sh"

# Patch hooks: disable local mining (we use remote MCP)
if grep -q "mempalace mine" "$HOOKS_DIR/mempal_save_hook.sh" 2>/dev/null; then
    python3 -c "
import re
for name in ['mempal_save_hook.sh', 'mempal_precompact_hook.sh']:
    path = '$HOOKS_DIR/' + name
    try:
        with open(path) as f:
            content = f.read()
        # Comment out any mempalace mine calls
        content = re.sub(r'^(\s*)(mempalace mine .*)$', r'\1# DISABLED for remote: \2', content, flags=re.MULTILINE)
        with open(path, 'w') as f:
            f.write(content)
    except FileNotFoundError:
        pass
" 2>/dev/null
fi
echo "  ✓ Hooks installed (save every 15 msgs + pre-compact)"

# 5. Wire hooks into settings.json
if [ -f "$SETTINGS" ]; then
    python3 -c "
import json

with open('$SETTINGS') as f:
    s = json.load(f)

hooks = s.setdefault('hooks', {})

# Add Stop hook if not already there
stop_cmd = '$HOOKS_DIR/mempal_save_hook.sh'
has_stop = any(
    any(h.get('command', '') == stop_cmd for h in entry.get('hooks', []))
    for entry in hooks.get('Stop', [])
)
if not has_stop:
    hooks.setdefault('Stop', []).append({
        'hooks': [{'type': 'command', 'command': stop_cmd, 'timeout': 30}]
    })

# Add PreCompact hook if not already there
precompact_cmd = '$HOOKS_DIR/mempal_precompact_hook.sh'
has_precompact = any(
    any(h.get('command', '') == precompact_cmd for h in entry.get('hooks', []))
    for entry in hooks.get('PreCompact', [])
)
if not has_precompact:
    hooks.setdefault('PreCompact', []).append({
        'hooks': [{'type': 'command', 'command': precompact_cmd, 'timeout': 30}]
    })

with open('$SETTINGS', 'w') as f:
    json.dump(s, f, indent=2)
    f.write('\n')
" 2>/dev/null && echo "  ✓ Hooks wired into settings.json"
fi

echo ""
echo "Done! Restart Claude Code to connect."
echo ""
echo "Verify with: claude mcp list | grep mempalace"
echo "Should show: mempalace: https://mp.thinks.bot/mcp (HTTP) - ✓ Connected"
