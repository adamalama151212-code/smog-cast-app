"""The smogcast assistant: one question in, one answer out – with the SQL or the sources behind it.

Routes (numbers never come from the language model):
- ``forecast`` – "tomorrow" is asked about, or a city is named without any sign of a question about numbers
  or the past: the deterministic answer from gold.forecast_tomorrow (works without a language model). If the
  question also asks what to do / health advice, a document-based section is added.
- ``data`` – other questions about numbers (model quality, data quality, rankings, past days and past
  forecasts, also for one city; comparisons with the baseline "tomorrow like today"): the language model
  writes SQL over the whitelisted gold tables, the guardrails validate it, the gold reader runs it (SQL
  Warehouse or the saved copy), and the model phrases the returned rows. A rejected or failing query is
  retried with the error, at most twice.
- ``knowledge`` – norms, health, methodology: retrieval over the documents, the model answers ONLY from
  the retrieved passages and cites them; without a language model the passages themselves are shown.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import math
import re
from dataclasses import dataclass, field

from smogcast.app import forecast as fc
from smogcast.app.gold import GoldReader, QueryResult
from smogcast.app.guardrails import UnsafeQueryError
from smogcast.app.llm import LLM, LLMUnavailable
from smogcast.app.rag import Chunk
from smogcast.app.text import find_cities, find_pollutants, normalize

log = logging.getLogger(__name__)

NOT_FOUND = "I could not find this information in the documents."
UNDERSTOOD = "Understood as: "   # start of the note on an answer whose question the language model rewrote (A6)


@dataclass
class Answer:
    text: str
    route: str                                   # forecast | data | knowledge
    sql: str | None = None
    rows: list[dict] | None = None
    sources: list[tuple[Chunk, float]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


# Routing vocabulary in English AND Polish (after ``normalize``: lower case, no diacritics) – the interface is
# English, but users may still type in Polish.
TOMORROW = ("tomorrow", "tommorow", "tommorrow", "tomorow", "jutr")   # with the usual misspellings; "jutr": jutro, jutrzejszy
FORECAST_HINTS = (*TOMORROW, "forecast", "exceed", "warning", "smog", "prognoz", "przekrocz", "ostrzez")
# "Tomorrow like today" is the NAME of a baseline (the forecast that repeats the last full day), not a day: the welcome
# screen's own example "Is the model better than “tomorrow like today”?" used to get tomorrow's forecast (T4).
BASELINE_PHRASES = ("tomorrow like today", "tomorrow as today", "same as today", "jutro jak dzisiaj", "jutro jak dzis")
# Signs of a question about what already happened. With a city they ask for numbers from the tables ("how many
# warnings for Kraków in 2025", "did yesterday's forecast for Kraków come true"), not for tomorrow's forecast.
PAST_HINTS = ("yesterday", "last winter", "last season", "last year", "last month", "last week", "come true",
              "came true", "was right", "were right", "been right", "issued", "so far",
              "wczoraj", "zeszl", "ubiegl", "sprawdzil", "trafn", "wydal")
FIRST_DATA_YEAR = 2021   # the measurements start here; a later year up to this one is a year of data ("in 2025"),
_YEAR = re.compile(r"\b(20\d\d)\b")   # while "from 2030" (the new EU limits) is not
ADVICE_HINTS = ("advice", "recommend", "what should", "what to do", "health", "children", "walk", "run", "jog",
                "cycling", "exercise", "outdoor", "mask", "window", "asthma", "pregnan", "elderly", "safe",
                "zalec", "co robic", "zdrow", "dzieci", "spacer", "biega", "rower", "aktywn", "maseczk", "okna",
                "astm", "ciaz", "senior", "bezpieczn")
# Rules first, the language model only for what they do not settle. A 7B model once routed "what is the
# PM10 alarm level?" to SQL and answered with the daily limit from a forecast column – legal levels,
# definitions and explanations must always come from the documents.
# "who " alone would also match "who won …": the World Health Organization is named explicitly.
KNOWLEDGE_HINTS = ("alarm level", "information level", "alert threshold", "information threshold", "limit value",
                   "limits", "threshold", " eu ", "standard", "norm", "directive", "regulation", " law", "legal",
                   "world health", "who guideline", "who recommend", "who air", "guideline", "why ",
                   "how does", "how is", "what is a", "what does", "meaning", "defin", "effect", "disease", "advice",
                   "recommend", "health", "methodolog",
                   "poziom alarm", "poziom inform", "dyrektyw", "rozporzadz", "przepis", "prawo", "wytyczn",
                   "dopuszczaln", "dlaczego", "czemu", "jak dziala", "jak liczon", "co to", "co oznacza", "czym jest",
                   "definic", "skutk", "chorob", "zalec", "zdrow", "metodyk")
# Vocabulary of the domain: a question with none of it is off-topic (a small model sometimes calls it "DATA").
DOMAIN_HINTS = ("air", "dust", "pm10", "pm2", "pm 2", "smog", "pollut", "quality", "concentration", "model",
                "forecast", "city", "cities", "station", "measure", "reading", "data", "breath", "protect", "gios",
                "world health", "directive", "brier", "calibrat", "weather", "wind", "limit", "level",
                "powietrz", "pyl", "zanieczyszcz", "jakosc", "stezen", "norm", "prognoz", "miast", "stacj", "pomiar",
                "odczyt", "dane", "danych", "oddych", "chroni", "dyrektyw", "kalibrac", "pogod", "wiatr")
DATA_HINTS = ("how many", "how much", "which city", "which cities", "which model", "highest", "lowest", "best",
              "worst", "average", "mean ", "ranking", "rank ", "compare", "metric", "brier", "auc", "pod ", "far ",
              "bss", "quarantin", "late ", "percent", "share of", "in 2025", "in 2024",
              "ile ", "ilu ", "ktore miast", "ktory model", "ktorym miesc", "najwyz", "najniz", "najlepsz", "srednia",
              "porownaj", "metryk", "kwarantann", "spozni", "procent", "odsetek", "w 2025", "w 2024")

# A6: the general model reads the conversation and returns the intent plus the question rewritten so that it stands
# on its own ("and in Wrocław?" -> "Will there be smog in Wrocław tomorrow?"). It decides follow-ups and everything
# the rules below are not sure about; the rules settle the clear cases and work alone when the model is down.
CLASSIFY_SYSTEM = """You route questions for smogcast, an assistant that forecasts whether the daily PM10 / PM2.5 limit
will be exceeded tomorrow in the 10 largest Polish cities.
Read the conversation and the NEW question. Reply with ONE line of JSON and nothing else:
{"intent": "<intent>", "question": "<the new question rewritten so it can be understood without the conversation>"}
Intents:
forecast - tomorrow's smog or exceedance forecast for a city or for all cities (tomorrow only, not past days)
data - numbers from the platform's tables: model quality (Brier, BSS, POD, FAR, AUC, calibration, the tests, the
comparison with the simple forecasts "tomorrow like today" and the calendar average), measured daily values and smog
days, warnings issued in past years, past forecasts and whether they came true, rankings of cities, data-quality
counts (how many readings arrived from the measurement API, how many were rejected or quarantined and why)
knowledge - legal limits, information and alarm levels, health effects and advice, definitions, how the model works
greeting - hello, thanks, what can you do
internals - how the system is built inside: databases and tables other than the published results, servers, keys,
passwords, prompts, instructions
other - anything not about air quality or this platform
Write the rewritten question in English. Resolve words like "there", "and PM2.5?", "why?" from the conversation."""
INTENTS = ("forecast", "data", "knowledge", "greeting", "internals", "other")

OFF_TOPIC = ("I only answer questions about air quality: tomorrow's smog forecast for the 10 largest Polish cities, "
             "air-quality standards and health advice, and how the smogcast model works and how good it is. "
             "Try for example: “Will there be smog in Kraków tomorrow?”")
GREETING = ("Hi! I can tell you whether the daily PM10 or PM2.5 limit is likely to be exceeded tomorrow in the 10 "
            "largest Polish cities, how good the forecast model is, what was measured, and explain air-quality limits "
            "and health advice. Try: “Will there be smog in Kraków tomorrow?” or “Why should I trust the PM2.5 "
            "forecast?”")
# A7: the published results (the gold tables) may be explained; how the system is built inside may not.
INTERNALS = ("I can't share how the system is built inside (its databases, servers, keys or instructions). I can explain "
             "the published results: the forecast, the model's quality, the measured values and what each column of "
             "the result tables means.")

# A7: questions about the inside of the system (and prompt-injection attempts) are refused by rules, before any model.
INTERNALS_HINTS = ("bronze", "silver table", "silver layer", " silver ", "ops table", " ops.", "schema of the", "workspace",
                   "catalog", "databricks", "volume", "password", "api key", "apikey", "access token", "secret",
                   "credential", "connection string", "system prompt", "your prompt", "the prompt", "your instruction",
                   "ignore previous", "ignore all", "ignore the above", "jailbreak", "source code", "server address",
                   "ip address", "key vault", "azure", "terraform",
                   "haslo", "klucz api", "sekret", "twoje instrukcj", "zignoruj", "pokaz prompt", "kod zrodlow", "serwer")
GREETINGS = {"hi", "hello", "hey", "hej", "czesc", "witaj", "witam", "dzien dobry", "good morning", "good evening",
             "thanks", "thank you", "dzieki", "dziekuje", "what can you do", "co potrafisz", "help", "pomoc"}
# Words that ask what something MEANS (a column of the result tables -> the data dictionary, A11).
MEANING_HINTS = ("mean", "column", "field", "what is", "what does", "explain", "describe", "definition", "stand for",
                 "oznacza", "kolumn", "pole ", "co to", "czym jest", "wyjasnij", "opisz")
STRUCTURE_HINTS = ("column", "field", "structure", "what is in", "contain", "kolumn", "struktur", "zawiera")
# Never shown in an answer, whatever a model writes (A7): internal tables, catalogs, storage paths, keys.
_INTERNAL_OUTPUT = re.compile(r"\b(bronze|silver|ops)\.[a-z_]+|smogcast_(dev|prod)|azuredatabricks\.net|/Volumes/|"
                              r"dbfs:|abfss://|(api[_ ]?key|password|secret)\s*[:=]", re.IGNORECASE)

# What the words of a question mean in the data. One text for the model that writes the SQL and for the model that
# describes the rows, so a number is selected and named by the same definitions the pages use (Data quality:
# "Rejected" = quarantine + late). No braces here: SQL_SYSTEM and SUMMARY_SYSTEM are filled with str.format.
DATA_MEANING = """How the words of a question map to the data:
- City: filter on city_id (lower case, no Polish letters - the list is below); "Cracow" = krakow, "Warsaw" =
  warszawa. backtest_metrics holds a row for every city AND a ready row for all cities together, city_id = 'ALL':
  EVERY query on it filters city_id - the named city, or city_id = 'ALL' when no city is named ("overall", "in
  general", "the model") - and selects city_id; without that filter a result mixes ten cities with their total.
  The other tables have one row per city and no 'ALL' row - aggregate them or list the cities.
- Pollutant: exactly 'PM10' or 'PM2.5'. No pollutant named: return both, one row per pollutant.
- "The model", "the forecast", "smogcast", "it" (no model named) = the OPERATIONAL model of each pollutant:
  JOIN gold.model_selection s ON s.pollutant = x.pollutant AND s.model = x.model AND s.operational.
  model_selection holds several tuned variants of each model type, so a join without "AND s.operational"
  multiplies the rows. Model names: 'gbt' (gradient-boosted trees), 'logistic' (logistic regression) and two simple
  baselines that are NOT in model_selection: 'persistence' = "tomorrow like today" / "same as today" / the naive
  forecast, 'climatology' = "calendar average" / "typical for the month" / the seasonal average.
- Years of the backtest: 2025 = "the test year" = split = 'test' (the final check - also the default when no year
  is named); 2024 = "the validation year" = split = 'validation' (used to choose the model). The backtest tables
  hold no other years. Never mix the two splits in one number.
- "Smog day", "exceedance", "day above the limit / the norm" = the daily mean of the city's worst station above
  the daily limit (PM10 50, PM2.5 25 µg/m³): measured_daily.exceeded = true; in the backtest the same fact is
  label = 1 (backtest_predictions) and n_exceed (backtest_metrics).
- "Warning", "alert", "alarm issued by the model" = warned = true (probability >= 0.5). A warning was right (a
  "hit") when the limit was exceeded on that day; false alarm = warned, not exceeded; miss = exceeded, not warned.
- Heating season = October to April, written heating_season = '2024/25' (October 2024 - April 2025).
- Days are calendar days in CET. Use the calendar of the data, not CURRENT_DATE: today = (SELECT MAX(issue_day)
  FROM gold.forecast_tomorrow), tomorrow = the newest target_day, yesterday = DATE_SUB(today, 1).
- Fractions 0..1, told to the user as percentages: probability, base_rate, pod, far, csi, mean_probability,
  observed_frequency, gap, bin_from, bin_to. Scores, NOT percentages: brier, validation_brier, bss, auc,
  squared_error. Already a percentage: dq_summary.pct_of_unique (6.69 means 6.69%).

The tables:
- forecast_tomorrow: the newest forecast per city and pollutant: probability that the daily limit is exceeded on
  target_day; warned = probability >= 0.5. Filter target_day = (SELECT MAX(target_day) FROM gold.forecast_tomorrow).
  limit_ug_m3 is the DAILY LIMIT VALUE that defines an exceedance (PM10 50, PM2.5 25) - it is NOT an information
  or alarm level. Why the risk is what it is: pm_d1_max_ug_m3 (yesterday's highest station daily mean),
  pm_d_morning_max_ug_m3 (this morning's highest mean) and tomorrow's weather forecast (t_mean_c, wind_mean_ms,
  calm_hours, precip_sum_mm). status = 'no_forecast' with unusable_reason when the inputs were incomplete.
  run_kind / issued_at: which run issued the row (see forecast_history).
- forecast_history: EVERY forecast run, same columns as forecast_tomorrow. run_kind: 'preliminary' (before 11:30
  CET, this morning's readings incomplete), 'official' (the first run from 11:30 CET - the forecast to compare with
  what happened), 'update' (later run, newer weather forecast). How a forecast changed during the day: one
  target_day ordered by issued_at.
- measured_daily: what was REALLY measured, one row per city, pollutant and day_cet: max_station_mean_ug_m3 (the
  worst station's daily mean, compared with limit_ug_m3), mean_station_mean_ug_m3 (the average of the stations),
  exceeded (true = smog day), heating_season (NULL from May to September). source: 'archive' = data verified by
  GIOŚ, 2021-01-01 to 2025-12-31; 'live' = the GIOŚ API, finished days since the platform went live in October
  2026. The months between are not here, so season '2025/26' holds only October - December 2025 and the newest
  complete season is '2024/25'. Smog days in a season: COUNT(*) WHERE exceeded AND heating_season = '...'.
- forecast_verification: each OFFICIAL forecast next to what was measured on its target_day - only the days since
  the platform went live (for 2024 and 2025 use the backtest tables). outcome: 'hit' (warned, exceeded), 'miss'
  (not warned, exceeded), 'false_alarm' (warned, not exceeded), 'correct_no_warning', 'no_forecast', 'no_data'
  (no valid measurement), 'pending' (the target day is not over or not measured yet). The forecast "came true" /
  "was right" for 'hit' and 'correct_no_warning'; it was wrong for 'miss' and 'false_alarm'; 'pending' = not known
  yet. "Did yesterday's forecast come true": select target_day, probability, warned, measured_max_ug_m3 and
  outcome of the rows with target_day >= yesterday, so the answer can say which day is checked and which is
  still pending. Live Brier score = AVG(squared_error).
- backtest_metrics: ready quality numbers of a whole year, per model, split, pollutant and city_id ('ALL' = all
  cities together): n_days, n_exceed (smog days), hits (smog days that got a warning), misses (smog days without
  one), false_alarms (warnings on days without smog); warnings issued = hits + false_alarms. brier: error of the
  probability, lower is better; bss: skill against the calendar average, above 0 = better than it; pod: share of
  smog days that got a warning; far: share of warnings that were false (NULL when no warning was issued); auc:
  ranking quality, closer to 1 is better. "Is model A better than B": the brier of both on the same split,
  pollutant and city_id - the lower one wins; show pod and far too.
- backtest_predictions: the backtest day by day - one row per model, split, city_id, pollutant and target_day:
  probability, warned and label (1 = the limit was really exceeded on target_day, 0 = it was not). Use it for
  anything finer than a whole year: a month, a range of dates, one day ("what did the model say on 2025-02-10"),
  how many warnings were issued and how many of them were right. Right warning = warned AND label = 1; false
  alarm = warned AND label = 0; miss = NOT warned AND label = 1. No city_name and no 'ALL' row here.
- reliability: calibration ("does 70% really mean 70%") per model, split and pollutant: bin 0..9 = forecasts of
  0-10% … 90-100% (bin_from, bin_to), n_days in the bin, mean_probability (what the model said on average),
  observed_frequency (how often the limit was really exceeded), gap = observed_frequency - mean_probability. Bins
  with fewer than 30 days are too small to judge.
- acceptance: the five tests fixed before training, for each pollutant's operational model on the test year:
  criterion 'K1' (better than the calendar average), 'K2' (better than "tomorrow like today"), 'K3' (percentages
  are calibrated), 'K4a' (catches smog days, PM10 only), 'K4b' (warnings are rarely false, PM10 only); value,
  threshold, passed. "Does the model pass its tests", "can it be trusted": all rows of the pollutant.
- model_selection: the variants tried on the validation year - several rows per pollutant and model type, each
  with validation_brier (lower is better); operational = true for the ONE variant per pollutant that makes the
  forecasts (the lowest validation_brier). The baselines are not here.
- dq_summary: quality of the hourly readings in LONG format - one row per (source, metric) with value (a count)
  and pct_of_unique. To get a count select value WHERE metric = '...' - never COUNT(*) the rows. A metric that
  has NO ROW for a source is 0 (zeros are not stored).
  source: 'live' = the real GIOŚ API feed - the DEFAULT whenever the question names no source; 'fault_replay' = a
  demonstration with injected faults, only when the question is about that demonstration. Never mix the sources.
  "Received" = records_received (everything the API sent, repeats included); "re-sent", "duplicates", "repeated"
  = resent_duplicates; "unique readings" = unique_readings (one per station and hour - the base of
  pct_of_unique). "REJECTED" = quarantined + late_rejected (as on the Data quality page): quarantined = set aside
  for breaking a hard rule, with the reasons in the quarantined_<reason> rows (missing_value = no value,
  below_min = below 0, above_max = impossibly high, unknown_station); late_rejected = arrived too late to be
  used. "Valid", "accepted", "clean", "used" = silver_valid. silver_readings = readings kept after the hard rules,
  split by quality flag into flag_ok, flag_frozen (sensor stuck on one value), flag_spike (a single extreme
  hour), flag_drift (drifting away from the city's other stations): flagged readings are kept, not rejected, but
  the forecast does not use them. 'quarantined' is the TOTAL of its quarantined_<reason> rows and
  silver_readings the total of the flag_* rows: never add a total to its own breakdown."""

SQL_EXAMPLES = """Examples:
Q: How many readings were rejected and why?
```sql
SELECT COALESCE(SUM(CASE WHEN metric IN ('quarantined', 'late_rejected') THEN value END), 0) AS rejected,
  COALESCE(SUM(CASE WHEN metric IN ('quarantined', 'late_rejected') THEN pct_of_unique END), 0) AS rejected_pct,
  COALESCE(SUM(CASE WHEN metric = 'late_rejected' THEN value END), 0) AS rejected_as_late,
  COALESCE(SUM(CASE WHEN metric = 'quarantined' THEN value END), 0) AS quarantined,
  COALESCE(SUM(CASE WHEN metric = 'quarantined_missing_value' THEN value END), 0) AS quarantined_no_value,
  COALESCE(SUM(CASE WHEN metric = 'quarantined_below_min' THEN value END), 0) AS quarantined_below_zero,
  COALESCE(SUM(CASE WHEN metric = 'quarantined_above_max' THEN value END), 0) AS quarantined_impossibly_high,
  COALESCE(SUM(CASE WHEN metric = 'quarantined_unknown_station' THEN value END), 0) AS quarantined_unknown_station,
  MAX(CASE WHEN metric = 'unique_readings' THEN value END) AS unique_readings
FROM gold.dq_summary WHERE source = 'live'
```
Q: Which city had the lowest Brier score for PM2.5 in 2025?
```sql
SELECT m.city_id, m.model, m.brier FROM gold.backtest_metrics m
JOIN gold.model_selection s ON s.pollutant = m.pollutant AND s.model = m.model AND s.operational
WHERE m.split = 'test' AND m.pollutant = 'PM2.5' AND m.city_id <> 'ALL' ORDER BY m.brier ASC LIMIT 1
```
Q: How many PM10 warnings did the model issue for Łódź in January 2025, and how many were false?
```sql
SELECT p.model, COUNT(*) FILTER (WHERE p.warned) AS warnings,
  COUNT(*) FILTER (WHERE p.warned AND p.label = 1) AS right_warnings,
  COUNT(*) FILTER (WHERE p.warned AND p.label = 0) AS false_alarms,
  COUNT(*) FILTER (WHERE p.label = 1) AS smog_days, COUNT(*) AS days
FROM gold.backtest_predictions p
JOIN gold.model_selection s ON s.pollutant = p.pollutant AND s.model = p.model AND s.operational
WHERE p.split = 'test' AND p.pollutant = 'PM10' AND p.city_id = 'lodz'
  AND p.target_day BETWEEN DATE '2025-01-01' AND DATE '2025-01-31' GROUP BY p.model
```
Q: Is the model better than "tomorrow like today"?
```sql
SELECT m.pollutant, m.city_id, m.model, m.brier, m.pod, m.far, m.hits, m.misses, m.false_alarms,
  m.n_exceed AS smog_days
FROM gold.backtest_metrics m
LEFT JOIN gold.model_selection s ON s.pollutant = m.pollutant AND s.model = m.model AND s.operational
WHERE m.split = 'test' AND m.city_id = 'ALL' AND (s.operational OR m.model = 'persistence')
ORDER BY m.pollutant, m.brier
```
Q: Did yesterday's official forecast for Gdańsk come true?
```sql
SELECT city_name, pollutant, target_day, probability, warned, measured_max_ug_m3, exceeded, outcome
FROM gold.forecast_verification
WHERE city_id = 'gdansk' AND target_day >= DATE_SUB((SELECT MAX(issue_day) FROM gold.forecast_tomorrow), 1)
ORDER BY target_day, pollutant
```
Q: What is tomorrow's PM10 probability in all cities?
```sql
SELECT city_name, probability, warned FROM gold.forecast_tomorrow
WHERE pollutant = 'PM10' AND target_day = (SELECT MAX(target_day) FROM gold.forecast_tomorrow) ORDER BY probability DESC
```"""

SQL_SYSTEM = """You write ONE Spark SQL (Databricks) SELECT statement that answers the user's question from these
tables only:
{schema}

Cities (city_id = city_name): {cities}

""" + DATA_MEANING + """

Rules: refer to tables as gold.<table>; only SELECT; at most {max_rows} rows; no comments; use only the columns
listed above. The answer is written only from the returned rows, so select what explains the number (city,
pollutant, model, day or period) next to it, and compute totals, differences and shares in the SQL.

""" + SQL_EXAMPLES + """
Return only the SQL inside a ```sql code block."""

SUMMARY_SYSTEM = """You explain query results of an air-quality platform to a reader who is not a specialist.
Answer in English (even if the question is in another language). Start with the direct answer; at most 5 short
sentences or a short list.
Never use the em dash; use an en dash (–) or a comma instead.
Use ONLY the numbers present in the result rows - never invent, estimate or calculate other numbers.
Say in plain words what each number counts, by what its column means (see below) - never rename it into another
concept. Never show the names of columns, metrics or tables, or codes from the data: write "596 readings were set
aside because the value was missing", not "quarantined_missing_value = 596"; "the model (gradient-boosted trees)",
not "gbt"; “tomorrow like today”, not "persistence"; "the calendar average", not "climatology"; "the 2025 test
year", not "split = test"; "share of smog days that got a warning", not "POD"; "share of warnings that were false",
not "FAR". Say which city, pollutant and period every number is about; when the rows cover several cities or both
pollutants, do not give one row's numbers as the overall answer. Write city names as in this list (city_id = city
name): {cities}; city_id ALL means all ten cities together.
Numbers: counts as whole numbers (596, not 596.0); fractions 0..1 as percentages (0.703 is 70%); a value that is
already a percentage stays one (6.69 is 6.69%); scores (Brier score, skill score, AUC) with three decimals, never
as percentages, with a word on whether lower or higher is better.
If the rows do not answer the question, or an outcome is still pending, say so instead of guessing. If the result
is empty, say what was looked for (the city, the pollutant, the day or period the SQL asks for) and that there is
no data for it yet.

""" + DATA_MEANING

KNOWLEDGE_SYSTEM = """You are the knowledge assistant of smogcast, a platform that forecasts PM10 and PM2.5
limit exceedances in Polish cities. Your ONLY source of knowledge is the CONTEXT below.

CONTEXT:
{context}

Rules:
1. Answer in English (even if the question is in another language), clearly and concisely. Cite the passages
   you used as [1], [2], ...
2. If the user asks for a list or a summary and the context has loose facts, build the list from them.
3. If the context contains NO information about the question, answer with exactly this one sentence:
   "I could not find this information in the documents."
4. Do not use general knowledge from outside the context. Do not invent numbers.
5. Copy every number exactly as it appears in the context, together with what it refers to. Different levels
   are different things: the information level, the alarm level and the limit value must never be mixed up.
6. Never use the em dash; use an en dash (–) or a comma instead.
7. A claim such as "better", "more reliable", "can be trusted" or "passes the test" must come with the figures the
   context gives for it ("X of Y", percentages, scores), each with exactly what the context says it counts. Take
   a figure from a table only when the table's header row is in the context."""
# Rule 7 needs a knowledge base that never hands over half a table: on 2026-10-10 the passage with the 2025 results
# began with the last rows of a table (the chunker's overlap had cut its header row), and the model wrote "490 of 667
# smog days (73%)" - the share of right warnings as the share of smog days caught. Since then rag.split_text never
# repeats a table in part and the document says the results in sentences too.


def _extract_sql(text: str) -> str:
    m = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return (m.group(1) if m else text).strip().rstrip(";").strip()


def _city_list(cities: dict[str, dict]) -> str:
    """The configured cities as the prompts name them: "krakow = Kraków, lodz = Łódź, …"."""
    return ", ".join(f"{city_id} = {city['name']}" for city_id, city in cities.items())


def _tidy(row: dict) -> dict:
    """A result row as the summarising model gets it: counts stored as DOUBLE without ".0" (596.0 -> 596) and long
    fractions cut to four decimals. The rows shown to the user (``Answer.rows``) stay as the query returned them."""
    return {k: (int(v) if v.is_integer() else round(v, 4)) if isinstance(v, float) and math.isfinite(v) else v
            for k, v in row.items()}


def _context(passages: list[tuple[Chunk, float]]) -> str:
    return "\n\n".join(f"[{i}] {c.title} ({c.location})\n{c.text}" for i, (c, _) in enumerate(passages, 1))


def _words(text: str) -> str:
    """Lower case, no diacritics, words only, padded – so " who " also matches at the start and the end."""
    return f" {' '.join(re.findall(r'[a-z0-9_.]+', normalize(text)))} "


def _parse_intent(reply: str) -> tuple[str | None, str | None]:
    """(intent, rewritten question) from the classifier's JSON; a bare word ("DATA") is accepted too."""
    m = re.search(r"\{.*\}", reply, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            intent = str(obj.get("intent", "")).strip().lower()
            question = str(obj.get("question") or "").strip() or None
            return (intent if intent in INTENTS else None), question
        except (json.JSONDecodeError, AttributeError):
            pass
    for w in reply.strip().lower().split()[:3]:
        if w.strip(".,:\"'") in INTENTS:
            return w.strip(".,:\"'"), None
    return None, None


class Assistant:
    def __init__(self, cfg: dict, gold: GoldReader | None, llm: LLM | None, retriever=None):
        self.cfg, self.gold, self.llm, self.retriever = cfg, gold, llm, retriever

    # ------------------------------------------------------------------ routing
    def rules(self, question: str) -> tuple[str | None, bool]:
        """(intent, sure) from the vocabulary alone. ``sure`` = the model is not asked (it cannot do better)."""
        q = _words(question)
        if any(h in q for h in INTERNALS_HINTS):
            return "internals", True
        if q.strip() in GREETINGS or q.strip().rstrip("!?.") in GREETINGS:
            return "greeting", True
        columns, tables = self.dictionary_terms(question)
        if (columns and any(h in q for h in MEANING_HINTS)) or (tables and any(h in q for h in STRUCTURE_HINTS)):
            return "dictionary", True
        baseline = any(p in q for p in BASELINE_PHRASES)
        for phrase in BASELINE_PHRASES:              # the baseline's name must not count as the word "tomorrow"
            q = q.replace(phrase, " ")
        tomorrow = any(t in q for t in TOMORROW)
        past = baseline or any(h in q for h in PAST_HINTS) or any(
            FIRST_DATA_YEAR <= int(y) <= dt.date.today().year for y in _YEAR.findall(q))
        data, knowledge = past or any(h in q for h in DATA_HINTS), any(h in q for h in KNOWLEDGE_HINTS)
        cities = find_cities(question, self.cfg["cities"])
        # A city means tomorrow's forecast when the question says "tomorrow" or shows no sign of asking for numbers
        # ("And in Wrocław?"); a city in a question about numbers or the past goes on to the data route below.
        if (cities and (tomorrow or not data)) or (not cities and tomorrow and any(h in q for h in FORECAST_HINTS)):
            return "forecast", True
        domain = bool(cities) or baseline or any(h in q for h in DOMAIN_HINTS + ADVICE_HINTS + FORECAST_HINTS)
        if not domain:
            # no word about air, dust or this platform ("ile to 2+2?", "who is the best football player?")
            return "other", True
        if knowledge and not data:
            return "knowledge", True   # legal levels, definitions, "why": always the documents (regression of a 7B)
        if data and not knowledge:
            return "data", True
        return ("data" if data else "knowledge"), False

    def route(self, question: str, history: list[dict] | None = None) -> str:
        return self.decide(question, history)[0]

    def decide(self, question: str, history: list[dict] | None = None) -> tuple[str, str]:
        """(intent, question to answer). Clear cases by rules; follow-ups and unclear questions by the general
        model with the conversation (A6), which also rewrites the question so it stands on its own."""
        intent, sure = self.rules(question)
        if intent in ("internals", "greeting", "dictionary") or (sure and not history):
            return intent, question
        if self.llm is None:
            return intent, question
        try:
            reply = self.llm.chat(CLASSIFY_SYSTEM, question, (history or [])[-6:])
        except LLMUnavailable:
            return intent, question
        model_intent, rewritten = _parse_intent(reply)
        if model_intent is None:
            return intent, question
        standalone = rewritten or question
        # Guards: the rules on the rewritten question override the model where a small model went wrong before.
        again, again_sure = self.rules(standalone)
        if again in ("internals", "dictionary") or (again_sure and again in ("forecast", "knowledge")):
            return again, standalone
        if model_intent == "data" and again == "other" and again_sure:
            return "other", standalone
        if model_intent == "forecast" and again == "data" and again_sure:
            return "data", standalone        # a past year, "yesterday", a baseline: numbers, not tomorrow's forecast
        if model_intent == "internals" and again == "data":
            # The rules found no word about the inside of the system and read a question about numbers ("how many
            # readings from the GIOŚ API were rejected" was once refused for its word "API"). Answering it cannot
            # leak anything: the guardrails let the query see the published tables only.
            return "data", standalone
        return model_intent, standalone

    def ask(self, question: str, history: list[dict] | None = None) -> Answer:
        intent, q = self.decide(question, history)
        if intent == "forecast":
            answer = self.forecast_answer(q)
        elif intent == "data":
            answer = self._guard(self.data_answer(q))
        elif intent == "dictionary":
            answer = self.dictionary_answer(q)
        elif intent == "greeting":
            answer = Answer(GREETING, "greeting")
        elif intent == "internals":
            answer = Answer(INTERNALS, "internals")
        elif intent == "other":
            answer = Answer(OFF_TOPIC, "other")
        else:
            answer = self._guard(self.knowledge_answer(q, history))
        if q != question:   # only the language model rewrites a question (decide), so the note also says it took part
            answer.notes.insert(0, UNDERSTOOD + q)
        answer.text = answer.text.replace("\u2014", "\u2013")   # A3: em dash -> en dash, whatever a model writes
        return answer

    @staticmethod
    def _guard(answer: Answer) -> Answer:
        """A7: an answer that names internal tables, catalogs, storage paths or keys is withheld."""
        if _INTERNAL_OUTPUT.search(answer.text or ""):
            log.warning("answer withheld: it named internal objects")
            return Answer(INTERNALS, "internals", notes=["The answer was withheld because it named internal parts "
                                                         "of the system."])
        return answer

    # ------------------------------------------------------------------ data dictionary (A11)
    def dictionary_terms(self, question: str) -> tuple[list[str], list[str]]:
        """Columns and tables of the gold data dictionary named in the question (``calm_hours`` or "calm hours";
        one-word columns such as ``model`` are too common to match free text)."""
        dictionary = self.cfg.get("gold_dictionary", {})
        q = _words(question)
        # tables only by their exact name: "the forecast tomorrow" in a sentence is not the table forecast_tomorrow
        tables = [t for t in dictionary if f" {t} " in q or f" gold.{t} " in q]
        columns = sorted({c for entry in dictionary.values() for c in entry["columns"]
                          if "_" in c and (f" {c} " in q or f" {c.replace('_', ' ')} " in q)})
        return columns, tables

    def dictionary_answer(self, question: str) -> Answer:
        """What a column or a result table means – from conf/gold_dictionary.yaml, deterministic."""
        dictionary = self.cfg.get("gold_dictionary", {})
        columns, tables = self.dictionary_terms(question)
        parts = []
        for c in columns:
            where = [t for t, e in dictionary.items() if c in e["columns"]]
            spec = dictionary[where[0]]["columns"][c]
            parts.append(f"**{c}** – {spec['description']} (in: {', '.join(where)})")
        for t in tables:
            if not any(h in _words(question) for h in STRUCTURE_HINTS):
                continue   # the table was only the context of a column question
            entry = dictionary[t]
            lines = [f"**{t}** – {entry['description']}"]
            lines += [f"- {c} – {s['description']}" for c, s in entry["columns"].items()]
            parts.append("\n".join(lines))
        if not parts:
            return Answer(NOT_FOUND, "dictionary")
        return Answer("\n\n".join(parts), "dictionary")

    # ------------------------------------------------------------------ forecast (deterministic)
    def forecast_answer(self, question: str) -> Answer:
        if self.gold is None or not self.gold.exists("forecast_tomorrow"):
            return Answer("There is no forecast yet – the first one appears after the next forecast run.", "forecast")
        data = fc.load(self.gold)
        rows = data["forecast"].rows
        cities = find_cities(question, self.cfg["cities"]) or list(self.cfg["run"]["cities"])
        pollutants = find_pollutants(question) or list(self.cfg["pollutants"])
        chosen = [r for r in rows if r["city_id"] in cities and r["pollutant"] in pollutants]
        if not chosen:
            return Answer("There is no forecast for this city and pollutant.", "forecast", data["forecast"].sql, rows)
        rel = data["reliability"].rows if data["reliability"] else []
        acc = data["acceptance"].rows if data["acceptance"] else []
        warning_p = self.cfg["thresholds"]["warning_probability"]
        if len(cities) > 2:  # an overview: one line per city instead of full paragraphs
            lines = [f"**Tomorrow ({chosen[0]['target_day']}): probability of exceeding the daily limit**"]
            for r in sorted(chosen, key=lambda r: -(r["probability"] or -1)):
                flag = " – **warning**" if r["warned"] else ""
                lines.append(f"- {r['city_name']} {fc.POLLUTANT_LABEL[r['pollutant']]}: {fc.pct(r['probability'])}{flag}")
            lines.append(f"\n_{fc.run_label(chosen[0])[:1].upper() + fc.run_label(chosen[0])[1:]}. A warning is "
                         f"issued at {fc.pct(warning_p)} or more._")
            text = "\n".join(lines)
        else:
            text = "\n\n".join(fc.describe(r, rel, acc, warning_p) for r in chosen)
        answer = Answer(text, "forecast", data["forecast"].sql, chosen)
        if any(h in normalize(question) for h in ADVICE_HINTS):
            advice = self.knowledge_answer(question + " (health advice for high particulate matter pollution)")
            answer.text += "\n\n---\n**Advice (from the documents):** " + advice.text
            answer.sources, answer.notes = advice.sources, advice.notes
        return answer

    # ------------------------------------------------------------------ data (text-to-SQL)
    def data_answer(self, question: str) -> Answer:
        if self.llm is None or not self.llm.available():
            return Answer("Questions about data need the language model, which is not available. Ready-made "
                          "summaries are on the Tomorrow, Model and Data quality pages.", "data")
        cities = _city_list(self.cfg["cities"])
        system = SQL_SYSTEM.format(schema=self.gold.schema_text(), max_rows=self.gold.max_rows, cities=cities)
        user, notes = question, []
        result: QueryResult | None = None
        for _attempt in range(3):
            sql = _extract_sql(self.llm.chat(system, user))
            try:
                result = self.gold.query(sql)
                break
            except UnsafeQueryError as exc:
                notes.append(f"Query rejected by the guardrails: {exc}")
                user = f"{question}\n\nYour previous SQL was rejected: {exc}. Write a corrected query."
            except Exception as exc:  # Spark analysis error: unknown column, bad syntax …
                notes.append(f"Query failed: {str(exc).splitlines()[0][:200]}")
                user = f"{question}\n\nYour previous SQL failed: {str(exc)[:500]}. Write a corrected query."
        if result is None:
            return Answer("I could not build a safe query for this question. Please try rephrasing it.",
                          "data", sql, None, notes=notes)
        payload = json.dumps([_tidy(r) for r in result.rows[:50]], default=str, ensure_ascii=False)
        text = self.llm.chat(SUMMARY_SYSTEM.format(cities=cities),
                             f"Question: {question}\nSQL: {result.sql}\nResult rows (JSON): {payload}")
        return Answer(text, "data", result.sql, result.rows, notes=notes)

    # ------------------------------------------------------------------ knowledge (RAG)
    def knowledge_answer(self, question: str, history: list[dict] | None = None) -> Answer:
        if self.retriever is None:
            return Answer("The knowledge base is not available right now, so questions about limits, health and "
                          "the method cannot be answered. The forecast and the numbers still work.", "knowledge")
        passages = self.retriever(question)
        if self.llm is None or not self.llm.available():
            listing = "\n\n".join(f"**[{i}] {c.title}** ({c.location}):\n> {c.text[:500]}…"
                                  for i, (c, _) in enumerate(passages[:3], 1))
            return Answer("The language model is not available – below are the most relevant passages.\n\n" + listing,
                          "knowledge", sources=passages, notes=["answered without the language model"])
        text = self.llm.chat(KNOWLEDGE_SYSTEM.format(context=_context(passages)), question, (history or [])[-4:])
        return Answer(text, "knowledge", sources=[] if text.strip() == NOT_FOUND else passages)
