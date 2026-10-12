# smogcast: tomorrow's PM10 and PM2.5 forecast for the 10 largest Polish cities

This document explains, for users of the smogcast assistant, what the forecast says, where its data comes from, how
the model was trained and tested, how good it is and where its limits are.

## 1. The question smogcast answers

> **Will the daily PM10 or PM2.5 limit be exceeded tomorrow in a city, and how sure is that?**

The answer is a **probability** from 0 to 100%, separately for PM10 and PM2.5, for each of the 10 largest cities in
Poland: Warszawa, Kraków, Wrocław, Łódź, Poznań, Gdańsk, Szczecin, Bydgoszcz, Lublin and Białystok. Some of them often
have smog (Kraków, Łódź), others are usually cleaner (Gdańsk, Szczecin).

The forecast is about tomorrow only: smogcast does not forecast several days ahead, and it does not forecast the
concentration itself (how many µg/m³), only whether the daily limit will be exceeded.

## 2. What the words mean

| Term | Meaning in smogcast |
|---|---|
| day | a calendar day in Central European Time (CET, UTC+1 all year), the way the Chief Inspectorate of Environmental Protection (GIOŚ) computes daily means |
| daily mean of a station | the mean of its hourly results; it counts only if at least 18 of the 24 hours were measured (the EU completeness rule) |
| PM10 exceedance | a daily mean **above 50 µg/m³**, the Polish and EU limit value for health protection (35 such days a year are allowed) |
| PM2.5 exceedance | a daily mean **above 25 µg/m³**, the EU daily limit value that applies from 2030 (Polish law has no daily PM2.5 limit today) |
| exceedance in a city | **at least one** station of the city has a valid daily mean above the limit; this is how GIOŚ reports exceedances |
| smog day | a day with an exceedance in a city |
| warning | a forecast of **50% or more** |
| "how sure" | the probability, checked for **calibration**: of all days forecast at about 70%, the limit should really be exceeded on about 70% |

## 3. Where the data comes from

- **History of measurements:** the yearly archive of GIOŚ with hourly PM10 and PM2.5 results from all stations,
  **2021–2025** (five heating seasons), about 11 million hourly values. These data are verified by GIOŚ and very
  clean. The model learned from these years and was tested on them.
- **Today's measurements:** the live data of GIOŚ, fetched every hour. Before they reach the forecast they are
  cleaned: repeated readings are merged, impossible values (below 0 or above 1000 µg/m³) and empty values are set
  aside, and suspicious readings (a sensor stuck on one value, a single extreme hour, a sensor slowly drifting away
  from the other stations of the city) are flagged and not used. Nothing is dropped without a reason.
- **Weather:** weather **forecasts** from Open-Meteo. The model always gets the weather forecast for tomorrow, never
  the weather that really happened, because the real weather is not known when the forecast is made.

## 4. How the forecast is made

Every day the model looks at what is known about the air and the weather, and gives the probability of an exceedance
tomorrow:
- **the air so far:** yesterday's daily mean (the worst station and the city average), whether yesterday was already
  above the limit, the day before yesterday, and this morning's readings (00:00–11:00 CET). Smog rarely appears or
  disappears overnight, so these are the strongest clues;
- **tomorrow's weather forecast:** temperature (cold days mean more home heating), wind (calm air, below 1.5 m/s,
  keeps smoke over the city), rain (it washes dust out), humidity, pressure, clouds and sunshine;
- **the calendar:** month, weekend, heating season (October–April);
- **the city:** each city has its own typical level of pollution.

The model is a set of **decision trees** (gradient-boosted trees). Each tree asks a few yes/no questions about these
inputs, for example "was yesterday above 40 µg/m³?" or "will there be more than 8 hours of calm wind?", and the trees
together give a probability between 0% and 100%.

The forecast for a day is computed **three times on the day before**:

| Run | Time (CET) | What it means |
|---|---|---|
| preliminary | about 00:30 | this morning's readings are not in yet |
| **official** | about 11:30 | the model has all its inputs; this is the forecast that is later checked against the measurements |
| update | about 17:30 | the same measurements, a newer weather forecast |

## 5. How the model was trained and tested

- **Training:** 2021–2023. **Choosing the model:** 2024. **Test:** 2025, a year the model never saw while learning.
  The years are split in time, never at random, because neighbouring days are alike.
- **The rules were fixed before training:** the model with the lowest error on 2024 is used, and five tests it must
  pass on 2025 were written down before any result was known. They were never changed after the results came in.
- **The test is honest:** in 2025 the model got the weather forecasts that really existed one day earlier, so the
  test shows how good the forecast is in real use.
- The results are reproducible: the same numbers come out on a laptop and in the cloud.

### The five tests

| Test | In plain words | Required |
|---|---|---|
| Better than the calendar average? | are the probabilities better than "how often the limit was exceeded in this city in this month in past years"? | yes |
| Better than "tomorrow like today"? | does the model add something to repeating the last full day? | yes |
| Does "70%" really mean about 70%? | in every group of days with a similar forecast (at least 30 days), forecast and reality may differ by at most 15 percentage points | yes |
| Does it catch smog days? (PM10 only) | share of smog days that got a warning the day before | at least 60% |
| Are its warnings rarely false? (PM10 only) | share of warnings that were followed by smog | at least half |

## 6. How we judge whether a forecast is good: coin toss, "tomorrow like today", the model

A forecast that warns about smog is judged from **two sides**, always both:
1. **Smog days that got a warning:** out of all smog days, how many were warned about the day before? (Did we miss
   smog?)
2. **Warnings followed by smog:** out of all warnings, how many were followed by a real smog day? (Can you trust a
   warning?)

Both numbers count the same days (a warning and smog on the same day), but divide them by something different: once
by all smog days, once by all warnings. A good forecast needs both to be high. Warning every single day would catch
every smog day, but most warnings would be wrong.

Two simple forecasts show how high the bar is:
- **A coin toss** (warn on heads): it catches about half of the smog days, but it also warns on half of the clean
  days, so only as many of its warnings are right as there are smog days in general: about 1 in 5 for PM2.5 and
  about 1 in 12 for PM10.
- **"Tomorrow like today"** (warn if the last full day was a smog day): a much stronger rival, because smog comes in
  episodes of several days. But it is always **two days late**. When the forecast is made in the morning, today is not
  over yet, so the last full day is yesterday; the forecast is for tomorrow. It therefore misses the first two days of
  every smog episode and keeps warning for two days after the smog is gone. It looks fine in long episodes (a frosty
  February), but when smog comes in short bursts of one or two days it never catches it.

**Example with one week**, smog from Monday to Thursday:

| | Mon | Tue | Wed | Thu | Fri | Sat | Sun |
|---|---|---|---|---|---|---|---|
| smog | yes | yes | yes | yes | no | no | no |
| "tomorrow like today" warns | | | yes | yes | yes | yes | |
| the model warns (example) | | yes | yes | yes | yes | | |

"Tomorrow like today" warned on 2 of the 4 smog days, and 2 of its 4 warnings were right. The model in this example
warned on 3 of the 4 smog days, and 3 of its 4 warnings were right: it also reads this morning's measurements and
the weather forecast (frost, no wind), so it can warn before an episode starts.

The real results of the test year 2025 follow, separately for PM2.5 and for PM10 (all 10 cities together; one smog
day = one city on one day).

### PM2.5 in the test year 2025: why the forecast can be trusted

| PM2.5 (697 smog days) | Smog days that got a warning | Warnings followed by smog |
|---|---|---|
| coin toss | about 50% | about 19% |
| "tomorrow like today" | 388 of 697 (56%) | 388 of 702 (55%) |
| **the model** | **490 of 697 (70%)** | **490 of 667 (73%)** |

In the test year 2025 the PM2.5 model warned on 490 of the 697 smog days (70%), and 490 of its 667 warnings were
followed by smog (73%). "Tomorrow like today" warned on 388 of the 697 smog days (56%), and 388 of its 702 warnings
were followed by smog (55%). A coin toss would catch about half of the smog days, but only about 19% of its warnings
would be right. **Why the PM2.5 forecast can be trusted:** the model beats both simple forecasts on both questions,
and it passes all the tests fixed before training.

February 2025 shows the same picture: the PM2.5 model caught 172 of 210 smog days with 29 false alarms, while
"tomorrow like today" caught 158 with 39 false alarms.

### PM10 in the test year 2025: why the forecast is less sure

| PM10 (292 smog days) | Smog days that got a warning | Warnings followed by smog |
|---|---|---|
| coin toss | about 50% | about 8% |
| "tomorrow like today" | 124 of 292 (42%) | 124 of 293 (42%) |
| **the model** | 129 of 292 (44%) | 129 of 189 (68%) |

In the test year 2025 the PM10 model warned on only 129 of the 292 smog days (44%), and 129 of its 189 warnings were
followed by smog (68%). "Tomorrow like today" warned on 124 of the 292 smog days (42%), and 124 of its 293 warnings
were followed by smog (42%). **Why the PM10 forecast is less sure:** the model's warnings are usually right, but it
catches fewer than half of the smog days, below the 60% that was required, so the PM10 model does not pass.

In February 2025 "tomorrow like today" caught a few more PM10 smog days than the model (56 vs 48 of 126), but it
raised 63 false alarms against the model's 19.

## 7. How good the forecast is (test year 2025)

| | PM10 | PM2.5 |
|---|---|---|
| days / smog days | 3613 / 292 (8%) | 3627 / 697 (19%) |
| error of the probability (Brier score, lower is better): model / calendar average / "tomorrow like today" | **0.043** / 0.062 / 0.093 | **0.075** / 0.117 / 0.172 |
| how well days are ranked from safest to most dangerous (AUC, 1 = perfect) | 0.95 | 0.94 |
| Better than the calendar average? | passed | passed |
| Better than "tomorrow like today"? | passed | passed |
| Does "70%" really mean about 70%? | passed, but only just (largest gap 8.5 percentage points; it was above the limit in nearly identical model versions) | passed (7 percentage points) |
| Does it catch smog days? | **not passed** (44%, required 60%) | not applicable |
| Are its warnings rarely false? | passed (32% false) | not applicable |
| **Verdict** | **does not pass all tests** | **passes all tests** |

- Smog is a matter of the **heating season**: almost all smog days were between November and March. From May to
  September there were almost no smog days, and the model correctly gave no warnings.
- 2025 was smoggier than 2024, and the model still did better than both simple forecasts in 9 of the 10 cities.

## 8. Why PM10 is harder to forecast than PM2.5

- PM2.5 is mostly **smoke** from home stoves and traffic. It depends mainly on the **weather**, which the model knows
  from the forecast: cold means more heating, calm wind keeps the smoke over the city.
- PM10 is PM2.5 **plus coarser dust**: raised from dry streets by traffic, from building sites, soil, and sand left
  after winter. These causes are not in the weather forecast, so the model sees only part of them.
- PM10 smog days are **rarer** (about 8% of days, against 19% for PM2.5), so the model has fewer examples to learn
  from, and the 50 µg/m³ limit needs a strong episode.
- More stations measure PM10, also next to busy streets. A local jump at one station is enough for a city exceedance,
  and such jumps cannot be seen in the weather forecast.

In short: the PM10 forecast is **less sure**, not useless. It ranks days well from safest to most dangerous and its
warnings are usually right, but it misses more than half of the smog days.

## 9. Live forecasts

Every day the forecast is made from the latest measurements and the latest weather forecast. A city without valid
measurements for yesterday, or without a complete weather forecast for tomorrow, gets "no forecast" with the reason,
never a guessed number. After each day ends, the official forecast is compared with what was really measured, so the
live record grows day by day. It is still short (live forecasts started in October 2026); the test year 2025 is the
main evidence of quality.

## 10. Limits

- The PM10 forecast does not pass all the tests (sections 6–8).
- Today's set of stations is slightly larger than in the history (new stations were opened). Because a city's value
  is its worst station, more stations can make today's values slightly higher than in the years the model learned from.
- The forecast uses measurements and the weather forecast only: no data on traffic, road conditions, building sites or
  pollution blown in from far away.
- The weather used for training was slightly more accurate than real day-ahead forecasts; the test used real day-ahead
  forecasts, so the test results are not inflated by this.
- PM2.5 uses the future EU daily limit (25 µg/m³, from 2030), because Polish law has no daily PM2.5 limit today.
