"""Text helpers for matching user input to configured cities."""

from __future__ import annotations

import re
import unicodedata


def normalize(name: str) -> str:
    """'Łódź' -> 'lodz'. The letters ł/Ł have no Unicode decomposition, so they are mapped by hand."""
    name = name.replace("ł", "l").replace("Ł", "L")
    decomposed = unicodedata.normalize("NFKD", name)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).strip().lower()


def resolve_city(cities: dict[str, dict], user_input: str) -> str | None:
    """Return the city_id whose id or display name matches the input (diacritics optional)."""
    wanted = normalize(user_input)
    for city_id, city in cities.items():
        if wanted in (normalize(city_id), normalize(city["name"])):
            return city_id
    return None


# Forms (after ``normalize``) under which users name the cities: English exonyms (Warsaw, Cracow) and Polish
# inflections, so questions work in both languages. Explicit lists, not stems: a stem like "pozna" would
# also match the Polish verb "poznać".
CITY_FORMS = {
    "warszawa": ["warsaw", "warszawa", "warszawie", "warszawy", "warszawe"],
    "krakow": ["krakow", "cracow", "krakowie", "krakowa", "krakowem"],
    "wroclaw": ["wroclaw", "wroclawiu", "wroclawia", "wroclawiem"],
    "lodz": ["lodz", "lodzi", "lodzia"],
    "poznan": ["poznan", "poznaniu", "poznania", "poznaniem"],
    "gdansk": ["gdansk", "gdansku", "gdanska", "gdanskiem"],
    "szczecin": ["szczecin", "szczecinie", "szczecina", "szczecinem"],
    "bydgoszcz": ["bydgoszcz", "bydgoszczy", "bydgoszcza"],
    "lublin": ["lublin", "lublinie", "lublina", "lublinem"],
    "bialystok": ["bialystok", "bialymstoku", "bialegostoku", "bialystoku", "bialymstokiem"],
}


def find_cities(text: str, cities: dict[str, dict]) -> list[str]:
    """Configured cities mentioned in a free-text question, in configuration order."""
    tokens = set(re.findall(r"[a-z]+", normalize(text)))
    return [c for c in cities if tokens & set(CITY_FORMS.get(c, [normalize(cities[c]["name"])]))]


def find_pollutants(text: str) -> list[str]:
    """PM10 / PM2.5 named in a question ("pm2,5", "pm 2.5", "pm25", "fine dust", "pył drobny" …); empty = not specified."""
    t = normalize(text).replace(" ", "")
    out = []
    if re.search(r"pm10(?![0-9])", t):
        out.append("PM10")
    if re.search(r"pm2[.,]?5", t) or any(w in t for w in ("pyldrobny", "finedust", "fineparticulate")):
        out.append("PM2.5")
    return out
