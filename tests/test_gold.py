"""The data layer: backends of the GoldReader, how the warehouse is signed in to (service principal or token), the
fallback to the saved copy, the configuration, and a guard that the data contract copied from the pipeline repo
stays unchanged."""

from __future__ import annotations

import datetime as dt
import hashlib
from pathlib import Path

import pytest

from smogcast.app import forecast as fc
from smogcast.app.config import load_config
from smogcast.app.gold import (SAMPLE_DIR, AutoBackend, DuckDBBackend, GoldReader, WarehouseBackend,
                               WarehouseSettings, service_principal_credentials, technical_columns)
from smogcast.app.guardrails import UnsafeQueryError

ROOT = Path(__file__).resolve().parents[1]


def test_warehouse_names_are_the_secure_views():
    wh = WarehouseBackend(WarehouseSettings("host", "/sql/1.0/warehouses/x", "token", "smogcast_prod"))
    reader = GoldReader(wh)
    sql = reader.physical("WITH last AS (SELECT MAX(target_day) d FROM gold.forecast_tomorrow) "
                          "SELECT f.city_id FROM gold.forecast_tomorrow f JOIN last ON f.target_day = last.d")
    assert sql.count("smogcast_prod.gold_secure.forecast_tomorrow") == 2 and "JOIN last" in sql   # CTE untouched


class Session:
    """What WarehouseBackend uses of a connector session; ``broken`` fails like an expired or restarted one."""

    description = [("n",)]

    def __init__(self, broken: bool = False):
        self.broken, self.closed = broken, False

    def cursor(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql):
        if self.broken:
            raise ConnectionError("session expired")

    def fetchall(self):
        return [(1,)]

    def close(self):
        self.closed = True


@pytest.fixture
def connector(monkeypatch):
    """``databricks.sql.connect`` replaced: records how each session signs in and hands out the prepared sessions."""
    from databricks import sql as dbsql

    class Connector:
        logins: list[dict] = []
        sessions: list[Session] = []

        def connect(self, **kwargs):
            self.logins.append(kwargs)
            return self.sessions.pop(0) if self.sessions else Session()

    fake = Connector()
    monkeypatch.setattr(dbsql, "connect", fake.connect)
    return fake


def test_the_hosted_app_signs_in_as_its_service_principal(connector):
    asked = []

    def credentials(settings):
        asked.append(settings.client_id)
        return "provider of the principal's tokens"

    settings = WarehouseSettings("host", "/sql/1.0/warehouses/x", "a token left in the shell", "smogcast_dev",
                                 client_id="app-id", client_secret="app-secret")
    assert WarehouseBackend(settings, credentials).run("SELECT 1 AS n") == [{"n": 1}]
    assert connector.logins == [{"server_hostname": "host", "http_path": "/sql/1.0/warehouses/x",
                                 "credentials_provider": "provider of the principal's tokens"}]   # no access_token
    assert asked == ["app-id"]
    assert "app-secret" not in repr(settings) and "a token left" not in repr(settings)   # never in a log line


def test_locally_the_short_lived_token_still_signs_in(connector):
    def never(settings):
        raise AssertionError("no service principal is configured")

    wh = WarehouseBackend(WarehouseSettings("host", "/sql/1.0/warehouses/x", "token", "smogcast_dev"), never)
    assert wh.run("SELECT 1 AS n") == [{"n": 1}]
    assert connector.logins == [{"server_hostname": "host", "http_path": "/sql/1.0/warehouses/x",
                                 "access_token": "token"}]


def test_a_broken_session_is_replaced_and_signs_in_again(connector):
    expired = Session(broken=True)
    connector.sessions[:] = [expired, Session()]
    settings = WarehouseSettings("host", "/sql/1.0/warehouses/x", "", "smogcast_dev", client_id="app-id",
                                 client_secret="app-secret")
    wh = WarehouseBackend(settings, lambda s: "provider")
    assert wh.run("SELECT 1 AS n") == [{"n": 1}]
    assert len(connector.logins) == 2 and expired.closed
    assert wh.run("SELECT 1 AS n") == [{"n": 1}] and len(connector.logins) == 2      # the new session is kept


def test_the_warehouse_is_configured_by_a_token_or_by_a_service_principal(monkeypatch):
    for name in ("DATABRICKS_TOKEN", "DATABRICKS_CLIENT_ID", "DATABRICKS_CLIENT_SECRET"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DATABRICKS_HOST", "https://adb-1.2.azuredatabricks.net/")
    monkeypatch.setenv("DATABRICKS_WAREHOUSE_ID", "abc")
    assert not WarehouseSettings.present()                        # an address alone: nobody to sign in as
    monkeypatch.setenv("DATABRICKS_CLIENT_ID", "app-id")
    assert not WarehouseSettings.present()                        # half a principal is none
    monkeypatch.setenv("DATABRICKS_CLIENT_SECRET", "app-secret")
    assert WarehouseSettings.present()
    settings = WarehouseSettings.from_env()
    assert settings.service_principal and settings.token == ""
    assert (settings.host, settings.http_path) == ("adb-1.2.azuredatabricks.net", "/sql/1.0/warehouses/abc")
    monkeypatch.delenv("DATABRICKS_CLIENT_ID")
    monkeypatch.delenv("DATABRICKS_CLIENT_SECRET")
    monkeypatch.setenv("DATABRICKS_TOKEN", "token")
    assert WarehouseSettings.present() and not WarehouseSettings.from_env().service_principal


def test_the_library_behind_the_service_principal_login_is_in_the_image():
    """Only the hosted app takes this path, so a missing library would show up there first. Nothing is asked of the
    network here: the provider signs in when the connector opens a session."""
    settings = WarehouseSettings("host", "/sql/1.0/warehouses/x", "", "smogcast_dev", client_id="app-id",
                                 client_secret="app-secret")
    assert callable(service_principal_credentials(settings))


def test_guardrails_run_before_any_backend():
    class Never:
        name = "never"

        def query(self, sql, allowed):
            raise AssertionError("a rejected query reached the backend")

    with pytest.raises(UnsafeQueryError):
        GoldReader(Never()).query("SELECT * FROM silver.pm_hourly")


def test_auto_falls_back_to_the_saved_copy_and_says_so():
    class Down:
        name = "warehouse"

        def ref(self, table):
            return f"c.gold_secure.{table}"

        def query(self, sql, allowed):
            raise ConnectionError("warehouse is asleep")

        columns = query

    copy = DuckDBBackend()
    copy.create("acceptance", "pollutant STRING, criterion STRING, passed BOOLEAN", [("PM10", "K1", True)])
    copy.available = lambda: True
    res = GoldReader(AutoBackend(Down(), copy)).query("SELECT * FROM gold.acceptance")
    assert res.source == "sample" and res.rows == [{"pollutant": "PM10", "criterion": "K1", "passed": True}]
    assert res.sql.startswith("SELECT * FROM gold.acceptance")           # the user sees gold.<table>, not the copy


def test_without_a_copy_the_warehouse_error_is_not_hidden():
    class Down:
        name = "warehouse"

        def query(self, sql, allowed):
            raise ConnectionError("warehouse is asleep")

    with pytest.raises(ConnectionError):
        GoldReader(AutoBackend(Down(), DuckDBBackend(None))).query("SELECT * FROM gold.acceptance")


def test_schema_text_leaves_out_the_columns_masked_for_the_app(cfg):
    """The app reads as an analyst: technical columns are always NULL there, so a model must not filter on them."""
    db = DuckDBBackend()
    db.create("dq_summary", "source STRING, metric STRING, value DOUBLE, computed_at TIMESTAMP", [])
    text = GoldReader(db, dictionary=cfg["gold_dictionary"]).schema_text()
    assert text.startswith("gold.dq_summary(source string, metric string, value double)")
    assert "computed_at" not in text


def test_schema_text_uses_spark_type_names_and_the_dictionary(cfg):
    db = DuckDBBackend()
    db.create("forecast_tomorrow", "city_id STRING, calm_hours BIGINT, issued_at TIMESTAMP", [])
    text = GoldReader(db, dictionary=cfg["gold_dictionary"]).schema_text()
    assert text.startswith("gold.forecast_tomorrow(city_id string, calm_hours bigint, issued_at timestamp)")
    assert "  - calm_hours: Forecast hours tomorrow with wind below 1.5 m/s" in text


@pytest.mark.skipif(not (SAMPLE_DIR / "forecast_tomorrow.json").exists(), reason="saved copy not present")
def test_saved_copy_reads_like_the_analyst_view(cfg):
    reader = GoldReader(DuckDBBackend(SAMPLE_DIR, technical_columns(cfg["gold_dictionary"])), max_rows=1000)
    data = fc.load(reader)
    rows = data["forecast"].rows
    assert len(rows) == 20 and data["forecast"].source == "sample"
    assert all(r["model_version"] is None and r["run_id"] is None for r in rows)   # technical columns masked (A16)
    assert isinstance(rows[0]["target_day"], dt.date) and isinstance(rows[0]["issued_at"], dt.datetime)
    assert "CET" in fc.run_label(rows[0])
    assert reader.exists("measured_daily") and not reader.exists("pm_hourly")


def test_configuration_has_the_shape_the_logic_reads(monkeypatch):
    monkeypatch.setenv("SMOGCAST_LLM_BASE_URL", "http://llm:1234/v1")
    cfg = load_config()
    assert cfg["run"]["cities"] == list(cfg["cities"]) and len(cfg["cities"]) == 10
    assert cfg["thresholds"]["warning_probability"] == 0.5 and cfg["pollutants"].keys() == {"PM10", "PM2.5"}
    assert cfg["app"]["llm"]["base_url"] == "http://llm:1234/v1"            # secrets and endpoints from the environment
    assert "forecast_tomorrow" in cfg["gold_dictionary"] and cfg["model"]["split"]["test_year"] == 2025


def test_data_contract_is_unchanged():
    """smogcast/app/COPIED.txt lists the data contract copied from the pipeline repo with its SHA-256
    (tools/copy_logic.py): cities, limits, model settings, the gold data dictionary and the model's input names.
    They are changed there and copied again, never edited here. The assistant's logic, conf/app.yaml and the
    knowledge base are this repo's own and must not be on the list."""
    lines = [ln.split() for ln in (ROOT / "smogcast/app/COPIED.txt").read_text(encoding="utf-8").splitlines()
             if ln and not ln.startswith("#")]
    changed = [name for digest, name in lines
               if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest]
    assert changed == [], changed
    assert sorted(name for _, name in lines) == [
        "smogcast/app/conf/cities.yaml", "smogcast/app/conf/gold_dictionary.yaml", "smogcast/app/conf/model.yaml",
        "smogcast/app/conf/thresholds.yaml", "smogcast/core/schema.py"]


def test_the_index_matches_the_documents():
    """The knowledge base is built from this repo's files (docs/rag, data/rag_sources, the gold data dictionary) by
    tools/build_index.py. A document or a dictionary entry changed without building again would leave the assistant
    quoting the old text: the passages cut from the files (no model needed) must be the ones in the index."""
    pytest.importorskip("pymupdf")
    from smogcast.app.rag import Index
    from smogcast.app.services import knowledge_chunks

    cfg = load_config()
    rag = cfg["app"]["rag"]
    missing = [s["file"] for s in rag["sources"] if not (ROOT / rag["sources_dir"] / s["file"]).exists()]
    assert missing == [], missing                                        # every source of app.yaml is in the repo
    index = Index.load(ROOT / "data" / "rag_index")
    assert index.model_name == rag["embedding_model"]
    assert knowledge_chunks(cfg) == index.chunks, "run: docker compose --profile tools run --rm index"
    assert len(index.vectors) == len(index.chunks)


def test_knowledge_question_finds_the_passages_without_a_language_model(cfg):
    """The real index (data/rag_index) and the real embedding model (in the image): a Polish question finds the
    passages about the PM10 alarm level, and without a language model the passages themselves are the answer."""
    pytest.importorskip("sentence_transformers")
    from smogcast.app.assistant import Assistant
    from smogcast.app.llm import LLMUnavailable
    from smogcast.app.services import load_retriever

    class Down:
        def available(self):
            return False

        def chat(self, *args, **kwargs):
            raise LLMUnavailable("no model in tests")

    retriever = load_retriever(cfg, ROOT / "data" / "rag_index")
    # no fixed number of passages: it changes with every edit of a document (test_the_index_matches_the_documents
    # is the one that keeps the index and the documents together)
    assert retriever is not None and retriever.index.chunks
    answer = Assistant(cfg, None, Down(), retriever).ask("Jaki jest poziom alarmowy dla PM10?")
    assert answer.route == "knowledge" and "not available" in answer.text and answer.sources
    # among the passages a language model would be given (top_k). Not "among the first three": a Polish question
    # shares only the token "PM10" with the English documents, so any passage that says PM10 often competes on
    # keywords (since 2026-10-12 the passage with the PM10 results of 2025 ranks third, the alarm level fourth).
    assert len(answer.sources) == cfg["app"]["rag"]["top_k"]
    assert any("150" in chunk.text for chunk, _ in answer.sources)
    english = retriever("What is the PM10 alarm level?")
    assert any("150" in chunk.text for chunk, _ in english[:2])            # in English it is found right away
