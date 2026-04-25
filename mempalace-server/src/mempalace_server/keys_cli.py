"""CLI for managing remote MCP API keys.

Usage:
    mempalace-api-key create <name> [--rate-limit N]
    mempalace-api-key list
    mempalace-api-key revoke <name>

DB connection comes from ``MEMPALACE_PGVECTOR_DSN``.
"""

from __future__ import annotations

import argparse
import hashlib
import secrets
import sys

import psycopg

from .deps import ensure_api_keys_table, get_dsn


def _generate_key() -> tuple[str, str]:
    """Return (raw_token, sha256_hash)."""
    token = "mp_live_" + secrets.token_hex(32)
    h = hashlib.sha256(token.encode("utf-8")).hexdigest()
    return token, h


def cmd_create(args: argparse.Namespace) -> int:
    ensure_api_keys_table()
    raw, key_hash = _generate_key()
    with psycopg.connect(get_dsn()) as conn:
        with conn.cursor() as cur:
            try:
                cur.execute(
                    "INSERT INTO mp_api_keys (name, key_hash, rate_limit) "
                    "VALUES (%s, %s, %s) RETURNING id, created_at",
                    (args.name, key_hash, args.rate_limit),
                )
                row = cur.fetchone()
            except psycopg.errors.UniqueViolation:
                conn.rollback()
                print(f"Error: an API key named '{args.name}' already exists.", file=sys.stderr)
                return 2
        conn.commit()
    print("API key created. Save this token — it will not be shown again:")
    print()
    print(f"  {raw}")
    print()
    print(f"  name:       {args.name}")
    print(f"  id:         {row[0]}")
    print(f"  rate_limit: {args.rate_limit}/min")
    print(f"  created_at: {row[1]}")
    return 0


def cmd_list(_args: argparse.Namespace) -> int:
    ensure_api_keys_table()
    with psycopg.connect(get_dsn()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT name, created_at, rate_limit, is_active FROM mp_api_keys "
                "ORDER BY created_at DESC"
            )
            rows = cur.fetchall()
    if not rows:
        print("No API keys.")
        return 0
    print(f"{'NAME':<24} {'CREATED':<28} {'RATE/MIN':<10} {'ACTIVE':<6}")
    for name, created_at, rate_limit, is_active in rows:
        print(f"{name:<24} {str(created_at):<28} {rate_limit:<10} {is_active!s:<6}")
    return 0


def cmd_revoke(args: argparse.Namespace) -> int:
    ensure_api_keys_table()
    with psycopg.connect(get_dsn()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE mp_api_keys SET is_active = false WHERE name = %s RETURNING id",
                (args.name,),
            )
            row = cur.fetchone()
        conn.commit()
    if row is None:
        print(f"No API key named '{args.name}'.", file=sys.stderr)
        return 2
    print(f"Revoked API key '{args.name}'.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mempalace-api-key",
        description="Manage MemPalace remote MCP API keys",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_create = sub.add_parser("create", help="Create a new API key")
    p_create.add_argument("name", help="Human-readable label for the key")
    p_create.add_argument(
        "--rate-limit",
        type=int,
        default=120,
        help="Requests per minute (default: 120)",
    )
    p_create.set_defaults(func=cmd_create)

    p_list = sub.add_parser("list", help="List all API keys")
    p_list.set_defaults(func=cmd_list)

    p_revoke = sub.add_parser("revoke", help="Revoke an API key by name")
    p_revoke.add_argument("name", help="Name of the API key to revoke")
    p_revoke.set_defaults(func=cmd_revoke)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
