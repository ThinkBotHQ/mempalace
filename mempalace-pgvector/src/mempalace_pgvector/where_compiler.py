"""Compile ChromaDB-style ``where`` dicts to parameterized PostgreSQL SQL.

MemPalace stores document metadata in a ``metadata JSONB`` column and the
drawer text in a ``document TEXT`` column. The core library passes filter
dicts that follow ChromaDB's ``where`` / ``where_document`` conventions
(RFC 001 §1.4). This module translates those dicts into a safe SQL fragment
and a list of positional parameters suitable for psycopg's ``execute()``.

Supported operators (parity with the ChromaDB reference backend):

Required:
    $eq, $ne, $in, $nin, $and, $or, $contains

Optional (implemented here because pgvector has no reason not to):
    $gt, $gte, $lt, $lte
"""

from __future__ import annotations

from typing import Any

from mempalace.backends.base import UnsupportedFilterError

__all__ = ["compile_where", "UnsupportedFilterError"]


_REQUIRED_OPERATORS = frozenset({"$eq", "$ne", "$in", "$nin", "$and", "$or", "$contains"})
_OPTIONAL_OPERATORS = frozenset({"$gt", "$gte", "$lt", "$lte"})
_SUPPORTED_OPERATORS = _REQUIRED_OPERATORS | _OPTIONAL_OPERATORS

_SCALAR_OPERATORS = frozenset({"$eq", "$ne", "$gt", "$gte", "$lt", "$lte"})
_LIST_OPERATORS = frozenset({"$in", "$nin"})
_DOCUMENT_ONLY_OPERATORS = frozenset({"$contains"})
_NUMERIC_OPERATORS = frozenset({"$gt", "$gte", "$lt", "$lte"})

_SQL_OP = {
    "$eq": "=",
    "$ne": "<>",
    "$gt": ">",
    "$gte": ">=",
    "$lt": "<",
    "$lte": "<=",
}


class _ParamCounter:
    __slots__ = ("_n",)

    def __init__(self, offset: int) -> None:
        if offset < 0:
            raise ValueError(f"param_offset must be >= 0, got {offset!r}")
        self._n = offset

    def next(self) -> str:
        self._n += 1
        return f"${self._n}"


def compile_where(
    where: dict | None,
    where_document: dict | None = None,
    param_offset: int = 0,
) -> tuple[str, list]:
    """Compile ``where`` and ``where_document`` dicts to SQL + params.

    Returns ``("", [])`` if both inputs are None/empty.
    The fragment does NOT include the leading ``WHERE`` keyword.
    """
    if not where and not where_document:
        return "", []

    counter = _ParamCounter(param_offset)
    params: list[Any] = []
    fragments: list[str] = []

    if where:
        if not isinstance(where, dict):
            raise UnsupportedFilterError(f"where must be a dict, got {type(where).__name__}")
        fragments.append(_compile_where_node(where, counter, params))

    if where_document:
        if not isinstance(where_document, dict):
            raise UnsupportedFilterError(
                f"where_document must be a dict, got {type(where_document).__name__}"
            )
        fragments.append(_compile_document_node(where_document, counter, params))

    if len(fragments) == 1:
        return fragments[0], params
    return "(" + " AND ".join(fragments) + ")", params


# ---------------------------------------------------------------------------
# Metadata (where) compilation
# ---------------------------------------------------------------------------


def _compile_where_node(node: dict, counter: _ParamCounter, params: list) -> str:
    if not isinstance(node, dict):
        raise UnsupportedFilterError(
            f"where clause must be a dict, got {type(node).__name__}: {node!r}"
        )
    if not node:
        raise UnsupportedFilterError("where clause must not be an empty dict")

    clauses: list[str] = []
    for key, value in node.items():
        if key.startswith("$"):
            clauses.append(_compile_logical(key, value, counter, params, document=False))
        else:
            clauses.append(_compile_field(key, value, counter, params))

    if len(clauses) == 1:
        return clauses[0]
    return "(" + " AND ".join(clauses) + ")"


def _compile_logical(
    op: str,
    value: Any,
    counter: _ParamCounter,
    params: list,
    *,
    document: bool,
) -> str:
    if op == "$and" or op == "$or":
        if not isinstance(value, list) or not value:
            raise UnsupportedFilterError(
                f"{op!r} requires a non-empty list of sub-clauses, got {value!r}"
            )
        joiner = " AND " if op == "$and" else " OR "
        compile_fn = _compile_document_node if document else _compile_where_node
        sub = [compile_fn(v, counter, params) for v in value]
        return "(" + joiner.join(sub) + ")"

    if op not in _SUPPORTED_OPERATORS:
        raise UnsupportedFilterError(
            f"operator {op!r} is not supported by the pgvector backend "
            f"(supported: {sorted(_SUPPORTED_OPERATORS)})"
        )
    raise UnsupportedFilterError(
        f"operator {op!r} cannot appear at the top level of a "
        f"{'where_document' if document else 'where'} clause; "
        f"wrap it in a field predicate like {{'field': {{{op!r}: ...}}}}"
    )


def _compile_field(field: str, value: Any, counter: _ParamCounter, params: list) -> str:
    if not isinstance(field, str) or not field:
        raise UnsupportedFilterError(f"metadata key must be a non-empty string, got {field!r}")

    if not isinstance(value, dict):
        return _compile_eq(field, value, counter, params, negate=False)

    if not value:
        raise UnsupportedFilterError(
            f"empty operator dict for field {field!r}; expected one of "
            f"{sorted(_SUPPORTED_OPERATORS)}"
        )

    clauses: list[str] = []
    for op, rhs in value.items():
        if not isinstance(op, str) or not op.startswith("$"):
            raise UnsupportedFilterError(
                f"expected an operator starting with '$' inside value for "
                f"field {field!r}, got key {op!r}"
            )
        if op not in _SUPPORTED_OPERATORS:
            raise UnsupportedFilterError(
                f"operator {op!r} is not supported by the pgvector backend "
                f"(supported: {sorted(_SUPPORTED_OPERATORS)})"
            )
        if op in _DOCUMENT_ONLY_OPERATORS:
            raise UnsupportedFilterError(
                f"operator {op!r} is only valid inside where_document, "
                f"not on metadata field {field!r}"
            )
        clauses.append(_compile_field_operator(field, op, rhs, counter, params))

    if len(clauses) == 1:
        return clauses[0]
    return "(" + " AND ".join(clauses) + ")"


def _compile_field_operator(
    field: str, op: str, rhs: Any, counter: _ParamCounter, params: list
) -> str:
    if op == "$eq":
        return _compile_eq(field, rhs, counter, params, negate=False)
    if op == "$ne":
        return _compile_eq(field, rhs, counter, params, negate=True)
    if op in _LIST_OPERATORS:
        return _compile_in(field, op, rhs, counter, params)
    if op in _NUMERIC_OPERATORS or op in _SCALAR_OPERATORS:
        return _compile_scalar_cmp(field, op, rhs, counter, params)
    raise UnsupportedFilterError(f"operator {op!r} cannot be applied directly to field {field!r}")


def _compile_eq(
    field: str,
    rhs: Any,
    counter: _ParamCounter,
    params: list,
    *,
    negate: bool,
) -> str:
    if isinstance(rhs, dict) or isinstance(rhs, list):
        raise UnsupportedFilterError(
            f"$eq / $ne on field {field!r} requires a scalar, got {type(rhs).__name__}: {rhs!r}"
        )

    if rhs is None:
        ph = counter.next()
        params.append(field)
        if negate:
            return f"(metadata ? {ph} AND jsonb_typeof(metadata->{ph}) <> 'null')"
        return f"(metadata ? {ph} AND jsonb_typeof(metadata->{ph}) = 'null')"

    if isinstance(rhs, bool):
        _escape_key(field)  # validate
        field_ph = counter.next()
        params.append(field)
        ph = counter.next()
        params.append(rhs)
        cmp = "<>" if negate else "="
        return f"((metadata->{field_ph})::jsonb {cmp} to_jsonb({ph}::boolean))"

    if isinstance(rhs, (int, float)):
        _escape_key(field)  # validate
        field_ph = counter.next()
        params.append(field)
        ph = counter.next()
        params.append(rhs)
        op = "<>" if negate else "="
        return f"((metadata->>{field_ph})::numeric {op} {ph}::numeric)"

    if isinstance(rhs, str):
        _escape_key(field)  # validate
        field_ph = counter.next()
        params.append(field)
        ph = counter.next()
        params.append(rhs)
        op = "<>" if negate else "="
        return f"(metadata->>{field_ph} {op} {ph})"

    raise UnsupportedFilterError(
        f"$eq / $ne on field {field!r} does not support value of type {type(rhs).__name__}: {rhs!r}"
    )


def _compile_in(field: str, op: str, rhs: Any, counter: _ParamCounter, params: list) -> str:
    if not isinstance(rhs, list):
        raise UnsupportedFilterError(
            f"{op!r} on field {field!r} requires a list, got {type(rhs).__name__}: {rhs!r}"
        )
    if not rhs:
        raise UnsupportedFilterError(f"{op!r} on field {field!r} requires a non-empty list")
    for item in rhs:
        if isinstance(item, (dict, list)):
            raise UnsupportedFilterError(
                f"{op!r} on field {field!r} requires scalar items, got "
                f"{type(item).__name__}: {item!r}"
            )

    _escape_key(field)  # validate

    all_numeric = all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in rhs)

    field_ph = counter.next()
    params.append(field)
    ph = counter.next()

    if all_numeric:
        params.append(list(rhs))
        if op == "$in":
            return f"((metadata->>{field_ph})::numeric = ANY({ph}::numeric[]))"
        return f"((metadata->>{field_ph})::numeric <> ALL({ph}::numeric[]))"

    params.append([_coerce_text(x) for x in rhs])
    if op == "$in":
        return f"(metadata->>{field_ph} = ANY({ph}::text[]))"
    return f"(metadata->>{field_ph} <> ALL({ph}::text[]))"


def _compile_scalar_cmp(field: str, op: str, rhs: Any, counter: _ParamCounter, params: list) -> str:
    # String comparisons (e.g. ISO date strings) work lexicographically and
    # are the canonical way ``tool_timeline`` filters by ``occurred_at``. We
    # check this BEFORE the numeric guard so a date string like
    # ``"2026-04-20"`` doesn't get rejected as non-numeric.
    _escape_key(field)  # validate

    if isinstance(rhs, str):
        sql_op = _SQL_OP[op]
        field_ph = counter.next()
        params.append(field)
        ph = counter.next()
        params.append(rhs)
        return f"(metadata->>{field_ph} {sql_op} {ph})"

    if isinstance(rhs, bool) or not isinstance(rhs, (int, float)):
        raise UnsupportedFilterError(
            f"{op!r} on field {field!r} requires a numeric or string value, got "
            f"{type(rhs).__name__}: {rhs!r}"
        )
    sql_op = _SQL_OP[op]
    field_ph = counter.next()
    params.append(field)
    ph = counter.next()
    params.append(rhs)
    return f"((metadata->>{field_ph})::numeric {sql_op} {ph}::numeric)"


# ---------------------------------------------------------------------------
# Document (where_document) compilation
# ---------------------------------------------------------------------------


def _compile_document_node(node: dict, counter: _ParamCounter, params: list) -> str:
    if not isinstance(node, dict):
        raise UnsupportedFilterError(
            f"where_document clause must be a dict, got {type(node).__name__}: {node!r}"
        )
    if not node:
        raise UnsupportedFilterError("where_document clause must not be an empty dict")

    clauses: list[str] = []
    for key, value in node.items():
        if key == "$contains":
            clauses.append(_compile_contains(value, counter, params))
        elif key in ("$and", "$or"):
            clauses.append(_compile_logical(key, value, counter, params, document=True))
        elif key.startswith("$"):
            if key in _SUPPORTED_OPERATORS:
                raise UnsupportedFilterError(
                    f"operator {key!r} is not valid inside where_document; "
                    f"only $contains, $and, $or are supported there"
                )
            raise UnsupportedFilterError(
                f"operator {key!r} is not supported by the pgvector backend"
            )
        else:
            raise UnsupportedFilterError(
                f"where_document does not accept bare field names; got {key!r}. "
                f"Use {{'$contains': <substring>}} to match against the document text"
            )

    if len(clauses) == 1:
        return clauses[0]
    return "(" + " AND ".join(clauses) + ")"


def _compile_contains(value: Any, counter: _ParamCounter, params: list) -> str:
    if not isinstance(value, str):
        raise UnsupportedFilterError(
            f"$contains requires a string value, got {type(value).__name__}: {value!r}"
        )
    if not value:
        raise UnsupportedFilterError("$contains requires a non-empty string")
    ph = counter.next()
    params.append(value)
    return f"(position({ph} IN document) > 0)"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _escape_key(key: str) -> str:
    if "\x00" in key:
        raise UnsupportedFilterError("metadata keys must not contain NUL bytes")
    if "'" in key or "\\" in key:
        raise UnsupportedFilterError(
            f"metadata key {key!r} contains characters not allowed in a "
            f"filter (single quote or backslash)"
        )
    return key


def _coerce_text(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        raise UnsupportedFilterError(
            "null is not a valid member of an $in / $nin list; use "
            "{'$eq': null} or {'$ne': null} on the field instead"
        )
    return str(value)
