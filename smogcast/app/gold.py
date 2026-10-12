"""Read-only access to gold tables for the assistant and the pages, through a SQL Warehouse (no Spark).

Every query – written by us or by a language model – goes through the same path:
1. ``validate_select`` (guardrails: one SELECT, whitelisted gold tables, enforced LIMIT),
2. table names written as ``gold.<table>`` are rewritten to the physical name of the backend,
3. execution by the backend. The SQL shown to the user is the validated query with ``gold.<table>`` names.

Backends:
- ``WarehouseBackend`` – the secure views ``<catalog>.gold_secure.<table>`` through a Databricks SQL Warehouse; the
  views apply row/column security for the app's identity (role ``analyst``, A16). Settings from the environment:
  the hosted app signs in as its service principal (OAuth), a developer with a short-lived token.
- ``DuckDBBackend`` – the same views as a saved copy (``tools/sample/<table>.json``) or test rows in an in-memory
  DuckDB, schema ``gold_secure``; Databricks SQL is translated by sqlglot. Technical columns are set to NULL, as the
  analyst sees them on the warehouse.
- ``AutoBackend`` – the warehouse, and the saved copy when the warehouse cannot be reached; each result says which
  one answered (``QueryResult.source``), so the interface can say when it shows the copy.

forecast.py and assistant.py use only ``QueryResult`` and ``GoldReader.exists / physical / query / schema_text``
(with ``max_rows`` and ``allowed``), so they do not know which backend answers.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import sqlglot
from sqlglot import exp

from smogcast.app.guardrails import DEFAULT_ALLOWED_TABLES, validate_select

log = logging.getLogger(__name__)

SECURE_SCHEMA = "gold_secure"
SAMPLE_DIR = Path(os.environ.get("SMOGCAST_SAMPLE_DIR", Path(__file__).resolve().parents[2] / "tools" / "sample"))
# Column types as a language model knows them from Spark SQL (DuckDB and information_schema spell them differently).
SPARK_TYPES = {"VARCHAR": "string", "STRING": "string", "INTEGER": "int", "INT": "int", "BIGINT": "bigint",
               "DOUBLE": "double", "FLOAT": "float", "BOOLEAN": "boolean", "DATE": "date", "TIMESTAMP": "timestamp",
               "TIMESTAMP_NTZ": "timestamp_ntz", "SMALLINT": "smallint", "TINYINT": "tinyint"}


@dataclass
class QueryResult:
    sql: str              # the validated query as written (gold.<table> names) – shown to the user
    rows: list[dict]
    source: str = ""      # which backend answered: "warehouse" | "sample"


def spark_type(name: str) -> str:
    return SPARK_TYPES.get(name.upper(), name.lower())


def rewrite(sql: str, ref: Callable[[str], str], allowed: frozenset[str]) -> str:
    """``gold.x`` / ``x`` → ``ref(x)`` for whitelisted tables (CTE names stay untouched)."""
    tree = sqlglot.parse_one(sql, read="databricks")
    ctes = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}

    def swap(node):
        if isinstance(node, exp.Table) and node.name.lower() in allowed and node.name.lower() not in ctes:
            new = sqlglot.parse_one(ref(node.name.lower()), into=exp.Table, read="databricks")
            if node.args.get("alias"):
                new.set("alias", node.args["alias"])
            return new
        return node

    return tree.transform(swap).sql(dialect="databricks")


def technical_columns(dictionary: dict) -> dict[str, list[str]]:
    """Columns of ``class: technical`` in the gold data dictionary – masked (NULL) for everyone but admins."""
    return {t: [c for c, spec in e.get("columns", {}).items() if spec.get("class") == "technical"]
            for t, e in (dictionary or {}).items()}


# --------------------------------------------------------------------------------------------- backends
@dataclass(frozen=True)
class WarehouseSettings:
    """Where the warehouse is and who asks. The hosted app signs in as its own service principal (role ``analyst``):
    ``DATABRICKS_CLIENT_ID`` + ``DATABRICKS_CLIENT_SECRET``, the secret from Key Vault. Locally the author's
    short-lived token does (``DATABRICKS_TOKEN`` from tools/dev-env.sh). The principal wins when both are set."""

    host: str
    http_path: str
    token: str = field(repr=False)
    catalog: str
    schema: str = SECURE_SCHEMA
    client_id: str = ""
    client_secret: str = field(default="", repr=False)

    @classmethod
    def from_env(cls) -> WarehouseSettings:
        env = os.environ
        return cls(host=env["DATABRICKS_HOST"].removeprefix("https://").rstrip("/"),
                   http_path=f"/sql/1.0/warehouses/{env['DATABRICKS_WAREHOUSE_ID']}",
                   token=env.get("DATABRICKS_TOKEN", ""), catalog=env.get("SMOGCAST_CATALOG", "smogcast_dev"),
                   client_id=env.get("DATABRICKS_CLIENT_ID", ""), client_secret=env.get("DATABRICKS_CLIENT_SECRET", ""))

    @staticmethod
    def present() -> bool:
        env = os.environ
        login = env.get("DATABRICKS_TOKEN") or (env.get("DATABRICKS_CLIENT_ID") and env.get("DATABRICKS_CLIENT_SECRET"))
        return bool(env.get("DATABRICKS_HOST") and env.get("DATABRICKS_WAREHOUSE_ID") and login)

    @property
    def service_principal(self) -> bool:
        return bool(self.client_id and self.client_secret)


def service_principal_credentials(settings: WarehouseSettings):
    """The connector's ``credentials_provider`` for OAuth machine to machine, as Databricks documents it for the SQL
    connector: the SDK exchanges the principal's id and secret for a token at the workspace (valid for an hour) and
    renews it by itself, so a session that lives for days never carries an expired one. ``auth_type`` is explicit
    because the SDK reads the environment too and refuses to choose when it also finds a token there."""
    from databricks.sdk.core import Config, oauth_service_principal

    def provider():
        return oauth_service_principal(Config(host=f"https://{settings.host}", client_id=settings.client_id,
                                              client_secret=settings.client_secret, auth_type="oauth-m2m"))

    return provider


class WarehouseBackend:
    name = "warehouse"

    def __init__(self, settings: WarehouseSettings | None = None, credentials=service_principal_credentials):
        self.settings = settings or WarehouseSettings.from_env()
        self._credentials = credentials   # tests pass their own: no network, no SDK
        self._columns: dict[str, list[tuple[str, str]]] | None = None
        self._conn = None
        self._lock = threading.Lock()

    def ref(self, table: str) -> str:
        return f"{self.settings.catalog}.{self.settings.schema}.{table}"

    def _connect(self):
        from databricks import sql as dbsql  # imported lazily: the app imports without a warehouse

        s = self.settings
        login = {"credentials_provider": self._credentials(s)} if s.service_principal else {"access_token": s.token}
        return dbsql.connect(server_hostname=s.host, http_path=s.http_path, **login)

    def run(self, sql: str) -> list[dict]:
        """One session reused across queries (opening one costs about a second); a broken session – expired, or
        the warehouse restarted – is replaced once before the error is passed on."""
        with self._lock:
            for attempt in (1, 2):
                if self._conn is None:
                    self._conn = self._connect()
                try:
                    with self._conn.cursor() as cur:
                        cur.execute(sql)
                        names = [c[0] for c in cur.description]
                        return [dict(zip(names, row)) for row in cur.fetchall()]
                except Exception:
                    conn, self._conn = self._conn, None
                    try:
                        conn.close()
                    except Exception:
                        pass
                    if attempt == 2:
                        raise
        raise RuntimeError("unreachable")

    def query(self, gold_sql: str, allowed: frozenset[str]) -> tuple[list[dict], str]:
        return self.run(rewrite(gold_sql, self.ref, allowed)), self.name

    def columns(self) -> dict[str, list[tuple[str, str]]]:
        """Columns of every view the identity can see, read once (information_schema shows only granted objects)."""
        if self._columns is None:
            s = self.settings
            rows = self.run(f"SELECT table_name, column_name, data_type FROM {s.catalog}.information_schema.columns "
                            f"WHERE table_schema = '{s.schema}' ORDER BY table_name, ordinal_position")
            out: dict[str, list[tuple[str, str]]] = {}
            for r in rows:
                out.setdefault(r["table_name"], []).append((r["column_name"], spark_type(r["data_type"])))
            self._columns = out
        return self._columns


class DuckDBBackend:
    """Gold views in an in-memory DuckDB: the saved copy (``directory``) or rows created by tests (``create``)."""

    name = "sample"

    def __init__(self, directory: Path | None = None, technical: dict[str, list[str]] | None = None):
        self.directory, self.technical = directory, technical or {}
        self._con = None
        self._lock = threading.Lock()

    def available(self) -> bool:
        return self.directory is not None and any(self.directory.glob("*.json"))

    @property
    def con(self):
        if self._con is None:
            import duckdb

            self._con = duckdb.connect()
            self._con.execute(f"CREATE SCHEMA {SECURE_SCHEMA}")
            if self.directory is not None:
                for path in sorted(self.directory.glob("*.json")):
                    self._load(path)
        return self._con

    def _load(self, path: Path) -> None:
        table = f"{SECURE_SCHEMA}.{path.stem}"
        self._con.execute(f"CREATE TABLE {table} AS SELECT * FROM read_json_auto('{path.as_posix()}', "
                          "maximum_object_size=67108864)")
        columns = {r[0]: r[1] for r in self._con.execute(f"DESCRIBE {table}").fetchall()}
        for column, kind in columns.items():
            if kind == "VARCHAR":
                self._restore_temporal(table, column)
        for column in self.technical.get(path.stem, []):
            if column in columns:
                self._con.execute(f"UPDATE {table} SET {column} = NULL")

    def _restore_temporal(self, table: str, column: str) -> None:
        """JSON keeps DATE and TIMESTAMP as ISO text; give them their types back, so ``issued_at + INTERVAL 1 HOUR``
        or ``target_day = DATE '…'`` work as on the warehouse. A column changes only if every value parses."""
        for kind, shape in (("DATE", "= 10"), ("TIMESTAMP", "> 10")):   # "2026-10-09" vs "2026-10-09T10:43:38…"
            bad = self._con.execute(f"SELECT count(*) FROM {table} WHERE {column} IS NOT NULL AND NOT "
                                    f"(length({column}) {shape} AND TRY_CAST({column} AS {kind}) IS NOT NULL)").fetchone()[0]
            filled = self._con.execute(f"SELECT count({column}) FROM {table}").fetchone()[0]
            if filled and not bad:
                self._con.execute(f"ALTER TABLE {table} ALTER {column} TYPE {kind} USING CAST({column} AS {kind})")
                return

    def create(self, table: str, schema: str, rows: list[tuple]) -> None:
        """A test table: ``schema`` in Spark DDL ("city_id STRING, probability DOUBLE, …"), DuckDB understands it."""
        with self._lock:
            self.con.execute(f"CREATE OR REPLACE TABLE {SECURE_SCHEMA}.{table} ({schema})")
            if rows:
                marks = ", ".join("?" * len(rows[0]))
                self.con.executemany(f"INSERT INTO {SECURE_SCHEMA}.{table} VALUES ({marks})", rows)

    def ref(self, table: str) -> str:
        return f"{SECURE_SCHEMA}.{table}"

    def run(self, sql: str) -> list[dict]:
        local = sqlglot.transpile(sql, read="databricks", write="duckdb")[0]
        with self._lock:   # one connection, used from worker threads one query at a time
            cur = self.con.execute(local)
            names = [c[0] for c in cur.description]
            return [dict(zip(names, row)) for row in cur.fetchall()]

    def query(self, gold_sql: str, allowed: frozenset[str]) -> tuple[list[dict], str]:
        return self.run(rewrite(gold_sql, self.ref, allowed)), self.name

    def columns(self) -> dict[str, list[tuple[str, str]]]:
        rows = self.run(f"SELECT table_name, column_name, data_type FROM information_schema.columns "
                        f"WHERE table_schema = '{SECURE_SCHEMA}' ORDER BY table_name, ordinal_position")
        out: dict[str, list[tuple[str, str]]] = {}
        for r in rows:
            out.setdefault(r["table_name"], []).append((r["column_name"], spark_type(r["data_type"])))
        return out


class AutoBackend:
    """The warehouse first; the saved copy when the warehouse cannot be reached (and the result says so)."""

    def __init__(self, primary, fallback):
        self.primary, self.fallback = primary, fallback

    @property
    def name(self) -> str:
        return (self.primary or self.fallback).name

    def ref(self, table: str) -> str:
        return (self.primary or self.fallback).ref(table)

    def _either(self, call):
        if self.primary is not None:
            try:
                return call(self.primary)
            except Exception:
                if not self.fallback.available():
                    raise
                log.warning("warehouse unavailable – answering from the saved copy", exc_info=True)
        return call(self.fallback)

    def query(self, gold_sql: str, allowed: frozenset[str]) -> tuple[list[dict], str]:
        return self._either(lambda b: b.query(gold_sql, allowed))

    def columns(self) -> dict[str, list[tuple[str, str]]]:
        return self._either(lambda b: b.columns())


def default_backend(dictionary: dict | None = None):
    """``SMOGCAST_DATA``: ``warehouse``, ``sample`` (the saved copy) or ``auto`` (default: the warehouse when it is
    configured, the copy otherwise and whenever the warehouse cannot be reached)."""
    mode = os.environ.get("SMOGCAST_DATA", "auto").strip().lower()
    sample = DuckDBBackend(SAMPLE_DIR, technical_columns(dictionary or {}))
    if mode == "sample":
        return sample
    if mode == "warehouse":
        return WarehouseBackend()
    return AutoBackend(WarehouseBackend() if WarehouseSettings.present() else None, sample)


# --------------------------------------------------------------------------------------------- the reader
class GoldReader:
    def __init__(self, backend=None, max_rows: int = 200, allowed: frozenset[str] = DEFAULT_ALLOWED_TABLES,
                 dictionary: dict | None = None):
        self.dictionary = dictionary or {}
        self.backend = backend if backend is not None else default_backend(self.dictionary)
        self.max_rows, self.allowed = max_rows, allowed

    def exists(self, table: str) -> bool:
        return table in self.backend.columns()

    def physical(self, sql: str) -> str:
        """``gold.x`` / ``x`` → the physical name of this backend (CTE names stay untouched)."""
        return rewrite(sql, self.backend.ref, self.allowed)

    def query(self, sql: str, max_rows: int | None = None) -> QueryResult:
        """``max_rows``: our own dashboard queries may read more rows than a language model is given."""
        safe = validate_select(sql, self.allowed, max_rows or self.max_rows)
        rows, source = self.backend.query(safe, self.allowed)
        return QueryResult(safe, rows, source)

    def schema_text(self) -> str:
        """Columns of every existing whitelisted table – the only schema a language model gets to see – with the
        meaning of each column from the gold data dictionary (conf/gold_dictionary.yaml, A11): a model that knows
        what `limit_ug_m3` means does not pass it off as an alarm level.

        Technical columns (``class: technical``) are left out: the app reads the secure views as an analyst (A16), for
        whom they are always NULL – a model that filtered on ``computed_at = MAX(computed_at)`` got no rows (stage 4,
        with Gemini)."""
        columns = self.backend.columns()
        masked = technical_columns(self.dictionary)
        lines = []
        for table in sorted(self.allowed):
            if table in columns:
                shown = [(c, t) for c, t in columns[table] if c not in masked.get(table, [])]
                lines.append(f"gold.{table}({', '.join(f'{c} {t}' for c, t in shown)})")
                described = self.dictionary.get(table, {}).get("columns", {})
                lines += [f"  - {c}: {described[c]['description']}" for c, _ in shown if c in described]
        return "\n".join(lines)
