"""SQL guardrails and city-name matching."""

import pytest

from smogcast.app.guardrails import UnsafeQueryError, validate_select
from smogcast.app.text import normalize, resolve_city


def test_select_gets_limit():
    out = validate_select("SELECT city_id FROM smogcast_prod.gold.forecast_tomorrow")
    assert out.endswith("LIMIT 1000")


def test_large_limit_is_capped():
    assert validate_select("SELECT * FROM gold.forecast_tomorrow LIMIT 999999").endswith("LIMIT 1000")


def test_local_path_table_in_gold_is_allowed():
    validate_select("SELECT * FROM delta.`/data/delta/gold/forecast_tomorrow`")


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE gold.forecast_tomorrow",
        "DELETE FROM gold.forecast_tomorrow",
        "SELECT 1; DROP TABLE gold.forecast_tomorrow",
        "SELECT * FROM silver.pm_hourly",                      # outside gold
        "SELECT * FROM silver.forecast_tomorrow",              # right name, wrong layer
        "SELECT * FROM delta.`/data/delta/silver/pm_hourly`",
        "SELECT * FROM gold.forecast_tomorrow UNION SELECT * FROM secrets",
        "SELECT * FROM (SELECT * FROM bronze.pm_hourly) t",    # subquery
        "INSERT INTO gold.forecast_tomorrow SELECT * FROM gold.forecast_tomorrow",
    ],
)
def test_unsafe_queries_are_blocked(sql):
    with pytest.raises(UnsafeQueryError):
        validate_select(sql)


@pytest.mark.parametrize("raw, norm", [("Łódź", "lodz"), ("KRAKÓW", "krakow"), (" Białystok ", "bialystok"), ("Gdańsk", "gdansk")])
def test_city_name_normalization(raw, norm):
    assert normalize(raw) == norm


def test_resolve_city_by_name_or_id():
    cities = {"lodz": {"name": "Łódź"}, "krakow": {"name": "Kraków"}}
    assert resolve_city(cities, "lodz") == "lodz"
    assert resolve_city(cities, "Kraków") == "krakow"
    assert resolve_city(cities, "Katowice") is None
