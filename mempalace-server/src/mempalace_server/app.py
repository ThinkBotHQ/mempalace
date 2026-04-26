"""Remote MCP server for MemPalace.

Wraps the existing in-process ``mempalace.mcp_server.TOOLS`` registry and
exposes it over Streamable HTTP at ``/mcp`` with bearer-token auth.

Run with::

    MEMPALACE_PGVECTOR_DSN=postgres://... mempalace-server

The first request from any client must include::

    Authorization: Bearer mp_live_<...>
"""

from __future__ import annotations

# --- 1. stdout protection bookkeeping -----------------------------------
# ``mempalace.mcp_server`` redirects fd 1 to fd 2 at import time so that
# noisy transitive dependencies cannot corrupt the JSON-RPC stream during
# stdio operation. We do NOT need that protection here (we are an HTTP
# server), so we restore stdout as soon as the import completes.
import os
import sys

_REAL_STDOUT = sys.stdout
_REAL_STDOUT_FD = None
try:
    _REAL_STDOUT_FD = os.dup(1)
except (OSError, AttributeError):
    _REAL_STDOUT_FD = None

# --- 2. Import the in-process server (this redirects stdout) ------------
from mempalace.mcp_server import TOOLS  # noqa: E402

# --- 3. Restore stdout immediately --------------------------------------
sys.stdout = _REAL_STDOUT
if _REAL_STDOUT_FD is not None:
    try:
        os.dup2(_REAL_STDOUT_FD, 1)
        os.close(_REAL_STDOUT_FD)
    except OSError:
        pass

import argparse  # noqa: E402
import contextlib  # noqa: E402
import inspect  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
from collections.abc import AsyncIterator  # noqa: E402
from typing import Any  # noqa: E402

import uvicorn  # noqa: E402
from mcp.server.lowlevel import Server as MCPServer  # noqa: E402
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager  # noqa: E402
from mcp.types import TextContent, Tool  # noqa: E402
from starlette.applications import Starlette  # noqa: E402
from starlette.requests import Request  # noqa: E402
from starlette.responses import JSONResponse  # noqa: E402
from starlette.routing import Mount, Route  # noqa: E402

from . import __version__  # noqa: E402
from .auth import BearerAuthMiddleware  # noqa: E402
from .deps import ensure_api_keys_table, shutdown_backend, warm_backend  # noqa: E402
from .rate_limit import RateLimiter  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("mempalace_server")


# ---------------------------------------------------------------------------
# Tool dispatch — replicates handle_request() in mempalace.mcp_server but
# adapted for the low-level MCP server's call_tool() interface.
# ---------------------------------------------------------------------------


def _coerce_args(tool_name: str, raw: dict[str, Any]) -> dict[str, Any]:
    """Whitelist + type-coerce arguments using the tool's input schema.

    Mirrors the logic in ``mempalace.mcp_server.handle_request`` so the HTTP
    transport behaves identically to the stdio transport.
    """
    spec = TOOLS[tool_name]
    schema_props = spec["input_schema"].get("properties", {})

    handler = spec["handler"]
    try:
        sig = inspect.signature(handler)
        accepts_var_keyword = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
    except (ValueError, TypeError):
        accepts_var_keyword = False

    args = dict(raw)
    if not accepts_var_keyword:
        args = {k: v for k, v in args.items() if k in schema_props}

    for key, value in list(args.items()):
        prop_schema = schema_props.get(key, {})
        declared_type = prop_schema.get("type")
        try:
            if declared_type == "integer" and not isinstance(value, int):
                args[key] = int(value)
            elif declared_type == "number" and not isinstance(value, (int, float)):
                args[key] = float(value)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid value for parameter '{key}'") from exc

    args.pop("wait_for_previous", None)
    return args


def _build_tool_list() -> list[Tool]:
    """Translate the in-process TOOLS dict to MCP ``Tool`` objects."""
    tools: list[Tool] = []
    for name, spec in TOOLS.items():
        tools.append(
            Tool(
                name=name,
                description=spec["description"],
                inputSchema=spec["input_schema"],
            )
        )
    return tools


def build_mcp_server() -> MCPServer:
    """Construct a low-level MCP Server with all 35 mempalace tools registered."""
    server: MCPServer = MCPServer(name="mempalace", version=__version__)

    @server.list_tools()
    async def _list_tools() -> list[Tool]:
        return _build_tool_list()

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        if name not in TOOLS:
            raise ValueError(f"Unknown tool: {name}")
        try:
            args = _coerce_args(name, arguments or {})
        except ValueError as exc:
            return [TextContent(type="text", text=f"Invalid arguments: {exc}")]

        handler = TOOLS[name]["handler"]
        try:
            # Existing handlers are synchronous. Run them in a thread so they
            # don't block the event loop on DB/embedding I/O.
            import anyio

            result = await anyio.to_thread.run_sync(lambda: handler(**args))
        except Exception:
            logger.exception("Tool error in %s", name)
            return [TextContent(type="text", text=f"Internal tool error in {name}")]

        text = json.dumps(result, indent=2, default=str)
        return [TextContent(type="text", text=text)]

    return server


# ---------------------------------------------------------------------------
# Starlette app assembly
# ---------------------------------------------------------------------------


async def _health(_request: Request) -> JSONResponse:
    return JSONResponse(
        {
            "status": "ok",
            "service": "mempalace-server",
            "version": __version__,
            "tools": len(TOOLS),
        }
    )


def build_app(*, dsn: str | None = None) -> Starlette:
    """Build the Starlette application.

    The HTTP layout is::

        GET  /health        — unauthenticated, monitoring probe
        ANY  /mcp           — Streamable HTTP MCP transport (auth required)
    """
    mcp_server = build_mcp_server()
    session_manager = StreamableHTTPSessionManager(
        app=mcp_server,
        event_store=None,
        json_response=True,
        stateless=True,
    )

    async def handle_mcp(scope, receive, send) -> None:
        await session_manager.handle_request(scope, receive, send)

    @contextlib.asynccontextmanager
    async def lifespan(_app: Starlette) -> AsyncIterator[None]:
        # 1. Ensure DB table exists for API keys.
        try:
            ensure_api_keys_table(dsn)
        except Exception:
            logger.exception("Failed to ensure mp_api_keys table")
            raise
        # 2. Warm the mempalace backend so the first MCP call is fast.
        try:
            warm_backend()
        except Exception:
            logger.exception("Backend warm-up failed; continuing anyway")
        # 3. Start the streamable HTTP session manager.
        async with session_manager.run():
            logger.info("mempalace-server %s ready: %d tools available", __version__, len(TOOLS))
            try:
                yield
            finally:
                shutdown_backend()

    rate_limiter = RateLimiter()

    async def _handle_mcp_route(request: Request):
        await handle_mcp(request.scope, request.receive, request._send)

    app = Starlette(
        debug=False,
        lifespan=lifespan,
        routes=[
            Route("/health", _health, methods=["GET"]),
            Route("/healthz", _health, methods=["GET"]),
            Mount("/mcp/", app=handle_mcp),
            Mount("/mcp", app=handle_mcp),
        ],
    )
    app.add_middleware(BearerAuthMiddleware, dsn=dsn, rate_limiter=rate_limiter)
    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mempalace-server",
        description="Remote MCP server for MemPalace (Streamable HTTP)",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("MEMPALACE_SERVER_PORT", "15033")),
        help="Bind port (default: 15033)",
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("MEMPALACE_SERVER_LOG_LEVEL", "info"),
        help="uvicorn log level (default: info)",
    )
    args = parser.parse_args(argv)

    if not os.environ.get("MEMPALACE_PGVECTOR_DSN"):
        print(
            "ERROR: MEMPALACE_PGVECTOR_DSN env var is required.",
            file=sys.stderr,
        )
        return 2

    app = build_app()
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        access_log=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
