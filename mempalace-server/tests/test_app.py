"""Smoke tests for the FastMCP/Starlette app construction.

These tests do NOT require a live Postgres — they only check that the
package imports cleanly and that the tool registry is mirrored.
"""

from __future__ import annotations


def test_imports_without_pgvector_env():
    """Importing the package must not raise even with no DSN set."""
    from mempalace_server import __version__

    assert __version__ == "0.1.0"


def test_tools_dict_imported():
    """We should see all 35 mempalace tools."""
    from mempalace_server.app import TOOLS

    # mempalace 3.3.x ships ~35 tools — assert at least that it's non-empty
    # and has the well-known core tools.
    assert len(TOOLS) >= 30
    for required in (
        "mempalace_status",
        "mempalace_search",
        "mempalace_add_drawer",
        "mempalace_health",
    ):
        assert required in TOOLS, f"missing core tool: {required}"


def test_build_tool_list_has_expected_shape():
    from mempalace_server.app import _build_tool_list

    tools = _build_tool_list()
    assert len(tools) >= 30
    sample = tools[0]
    assert sample.name
    assert sample.description
    assert sample.inputSchema is not None


def test_coerce_args_whitelists_unknown():
    from mempalace_server.app import _coerce_args

    args = _coerce_args(
        "mempalace_search",
        {"query": "hello", "limit": "5", "evil": "ignore-me"},
    )
    assert args["query"] == "hello"
    assert args["limit"] == 5  # coerced from str
    assert "evil" not in args


def test_keys_cli_parser_builds():
    from mempalace_server.keys_cli import build_parser

    parser = build_parser()
    ns = parser.parse_args(["create", "demo", "--rate-limit", "60"])
    assert ns.cmd == "create"
    assert ns.name == "demo"
    assert ns.rate_limit == 60
