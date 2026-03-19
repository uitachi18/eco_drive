import json
import os
import re
import sqlite3
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any


VIN_RE = re.compile(r"\b([A-HJ-NPR-Z0-9]{17})\b", re.IGNORECASE)  # excludes I,O,Q
YEAR_RE = re.compile(r"\b(19[7-9]\d|20[0-3]\d)\b")

# Very small alias map to infer make from iconic model names when user omits make.
# This keeps things usable for inputs like "mustang 1980" without claiming full coverage.
MODEL_TO_MAKE_ALIASES: dict[str, str] = {
    "mustang": "ford",
    "corvette": "chevrolet",
    "camaro": "chevrolet",
    "civic": "honda",
    "accord": "honda",
    "supra": "toyota",
    "skyline": "nissan",
    "gt-r": "nissan",
    "gtr": "nissan",
    "911": "porsche",
    "prius": "toyota",
    "model 3": "tesla",
    "model s": "tesla",
    "model x": "tesla",
    "model y": "tesla",
}


@dataclass(frozen=True)
class VehicleProfile:
    source: str
    query: str
    fetched_at: float
    data: dict[str, Any]


def _default_cache_path() -> str:
    # keep it out of git and predictable across runs
    base = os.getenv("LOCALAPPDATA") or os.getenv("APPDATA") or os.path.expanduser("~")
    return os.path.join(base, "ecodrive_vehicle_cache.sqlite")


def _get_cache_path() -> str:
    return os.getenv("VEHICLE_CACHE_PATH", _default_cache_path())


def _connect() -> sqlite3.Connection:
    path = _get_cache_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS vehicle_cache (
          cache_key TEXT PRIMARY KEY,
          source TEXT NOT NULL,
          query TEXT NOT NULL,
          fetched_at REAL NOT NULL,
          json TEXT NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_vehicle_cache_fetched_at ON vehicle_cache(fetched_at)")
    return conn


def _cache_key(source: str, query: str) -> str:
    return f"{source}::{query.strip().lower()}"


def cache_get(source: str, query: str, max_age_days: int = 180) -> VehicleProfile | None:
    key = _cache_key(source, query)
    max_age_s = max(0, int(max_age_days)) * 86400
    now = time.time()
    with _connect() as conn:
        row = conn.execute(
            "SELECT source, query, fetched_at, json FROM vehicle_cache WHERE cache_key = ?",
            (key,),
        ).fetchone()
    if not row:
        return None
    fetched_at = float(row[2])
    if max_age_s and (now - fetched_at) > max_age_s:
        return None
    try:
        data = json.loads(row[3])
    except Exception:
        return None
    return VehicleProfile(source=row[0], query=row[1], fetched_at=fetched_at, data=data)


def cache_put(source: str, query: str, data: dict[str, Any]) -> VehicleProfile:
    key = _cache_key(source, query)
    fetched_at = time.time()
    payload = json.dumps(data, ensure_ascii=False)
    with _connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO vehicle_cache(cache_key, source, query, fetched_at, json) VALUES(?,?,?,?,?)",
            (key, source, query, fetched_at, payload),
        )
        conn.commit()
    return VehicleProfile(source=source, query=query, fetched_at=fetched_at, data=data)


def _http_get_json(url: str, timeout_s: int = 10) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "EcoDrive/1.0 (offline-vehicle-lookup)",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    return json.loads(raw)


def extract_vin(text: str) -> str | None:
    if not text:
        return None
    m = VIN_RE.search(text.upper())
    return m.group(1) if m else None


def parse_year_make_model(text: str) -> tuple[str, str, str] | None:
    """
    Parse flexible Y/M/M from free-form text.
    Supports:
      - "1980 Ford Mustang"
      - "Ford Mustang 1980"
      - "mustang 1980" (infers make for some iconic models)
    Returns (year, make, model) or None.
    """
    if not text:
        return None
    t = " ".join(str(text).strip().split())
    ym = YEAR_RE.search(t)
    if not ym:
        return None
    year = ym.group(1)

    remainder = (t[: ym.start()] + " " + t[ym.end() :]).strip()
    # Remove punctuation-ish separators
    remainder = re.sub(r"[,\(\)\[\]\{\}\|/]+", " ", remainder)
    remainder = " ".join(remainder.split())
    if not remainder:
        return None

    tokens = remainder.split(" ")
    tokens_l = [x.lower() for x in tokens if x]

    make = ""
    model = ""

    if len(tokens_l) >= 2:
        make = tokens_l[0]
        model = " ".join(tokens_l[1:])
    else:
        # single token -> assume model, infer make when possible
        model = tokens_l[0]
        make = MODEL_TO_MAKE_ALIASES.get(model, "")

    # Try a second inference pass for multi-word model aliases
    if not make:
        for alias, inferred_make in MODEL_TO_MAKE_ALIASES.items():
            if alias in remainder.lower():
                make = inferred_make
                model = alias  # use the alias as model string
                break

    if not (year and make and model):
        return None
    return (year, make, model)


def vpci_decode_vin(vin: str, use_cache: bool = True) -> VehicleProfile | None:
    """
    NHTSA vPIC VIN decode.
    Returns a normalized dict in VehicleProfile.data.
    """
    vin = (vin or "").strip().upper()
    if not VIN_RE.fullmatch(vin):
        return None

    source = "vpci_vin_decode"
    if use_cache:
        cached = cache_get(source, vin)
        if cached:
            return cached

    url = f"https://vpic.nhtsa.dot.gov/api/vehicles/DecodeVinValuesExtended/{urllib.parse.quote(vin)}?format=json"
    try:
        payload = _http_get_json(url)
        results = (payload or {}).get("Results") or []
        r0 = results[0] if results else {}
        # Keep only high-signal fields (vPIC returns lots of empties)
        keep = [
            "VIN",
            "Make",
            "Model",
            "ModelYear",
            "BodyClass",
            "VehicleType",
            "Series",
            "Trim",
            "EngineCylinders",
            "DisplacementL",
            "FuelTypePrimary",
            "DriveType",
            "TransmissionStyle",
            "TransmissionSpeeds",
            "PlantCountry",
            "PlantCompanyName",
            "PlantCity",
            "PlantState",
        ]
        data = {k: r0.get(k) for k in keep if r0.get(k)}
        if not data:
            return None
        return cache_put(source, vin, data) if use_cache else VehicleProfile(source, vin, time.time(), data)
    except Exception:
        return None


def vpci_models_for_make_year(make: str, year: str | int, use_cache: bool = True) -> VehicleProfile | None:
    """
    NHTSA vPIC: list models for a make in a given model year.
    Useful as a broad-coverage fallback when trim/spec APIs have gaps.
    """
    mk = (make or "").strip().lower()
    y = str(year).strip()
    if not (mk and y and YEAR_RE.fullmatch(y)):
        return None

    query = f"{y} {mk}"
    source = "vpci_models_for_make_year"
    if use_cache:
        cached = cache_get(source, query, max_age_days=365)
        if cached:
            return cached

    url = f"https://vpic.nhtsa.dot.gov/api/vehicles/GetModelsForMakeYear/make/{urllib.parse.quote(mk)}/modelyear/{urllib.parse.quote(y)}?format=json"
    try:
        payload = _http_get_json(url)
        results = (payload or {}).get("Results") or []
        if not results:
            return None
        models = sorted({(r.get("Model_Name") or "").strip() for r in results if r.get("Model_Name")})
        models = [m for m in models if m]
        if not models:
            return None
        data = {"make": mk, "year": y, "models": models}
        return cache_put(source, query, data) if use_cache else VehicleProfile(source, query, time.time(), data)
    except Exception:
        return None


def vpci_models_for_make(make: str, use_cache: bool = True) -> VehicleProfile | None:
    """
    NHTSA vPIC: list models for a make (all years).
    Broad coverage; useful when user does not provide a year.
    """
    mk = (make or "").strip().lower()
    if not mk:
        return None

    query = mk
    source = "vpci_models_for_make"
    if use_cache:
        cached = cache_get(source, query, max_age_days=365)
        if cached:
            return cached

    url = f"https://vpic.nhtsa.dot.gov/api/vehicles/getmodelsformake/{urllib.parse.quote(mk)}?format=json"
    try:
        payload = _http_get_json(url)
        results = (payload or {}).get("Results") or []
        if not results:
            return None
        models = sorted({(r.get('Model_Name') or '').strip() for r in results if r.get('Model_Name')})
        models = [m for m in models if m]
        if not models:
            return None
        data = {"make": mk, "models": models}
        return cache_put(source, query, data) if use_cache else VehicleProfile(source, query, time.time(), data)
    except Exception:
        return None


def parse_make_model(text: str) -> tuple[str, str] | None:
    """
    Parse "MAKE MODEL" without year.
    Very lightweight: first token is make, rest is model.
    """
    if not text:
        return None
    t = " ".join(str(text).strip().split())
    # If a year exists, prefer parse_year_make_model instead.
    if YEAR_RE.search(t):
        return None
    # Keep only word-ish characters and spaces
    t = re.sub(r"[,\(\)\[\]\{\}\|/]+", " ", t)
    t = " ".join(t.split())
    tokens = t.split(" ")
    if len(tokens) < 2:
        return None
    make = tokens[0].lower()
    model = " ".join(tokens[1:]).lower()
    return (make, model)


def carquery_search(year: str | int, make: str, model: str, use_cache: bool = True) -> VehicleProfile | None:
    """
    CarQuery API: returns trims/spec-like data when available.
    Coverage varies; treat as best-effort enrichment.
    """
    y = str(year).strip()
    mk = (make or "").strip()
    md = (model or "").strip()
    if not (y and mk and md):
        return None

    query = f"{y} {mk} {md}"
    source = "carquery_trims"
    if use_cache:
        cached = cache_get(source, query)
        if cached:
            return cached

    params = {
        "cmd": "getTrims",
        "year": y,
        "make": mk,
        "model": md,
    }
    url = f"https://www.carqueryapi.com/api/0.3/?{urllib.parse.urlencode(params)}"
    try:
        payload = _http_get_json(url)
        # CarQuery returns JSON with "Trims": [...]
        trims = (payload or {}).get("Trims") or []
        if not trims:
            return None
        # Normalize: include a small selection of fields from top trims
        pick_keys = [
            "model_id",
            "model_make_id",
            "model_name",
            "model_trim",
            "model_year",
            "model_body",
            "model_engine_position",
            "model_engine_cc",
            "model_engine_power_ps",
            "model_engine_torque_nm",
            "model_engine_fuel",
            "model_transmission_type",
            "model_drive",
            "model_seats",
            "model_doors",
            "model_weight_kg",
        ]
        normalized = []
        for t in trims[:10]:
            norm = {k: t.get(k) for k in pick_keys if t.get(k)}
            if norm:
                normalized.append(norm)
        if not normalized:
            return None
        data = {"query": query, "trims": normalized}
        return cache_put(source, query, data) if use_cache else VehicleProfile(source, query, time.time(), data)
    except Exception:
        return None


def format_vehicle_profile(profile: VehicleProfile) -> str:
    d = profile.data or {}
    if profile.source == "vpci_vin_decode":
        parts = []
        year = d.get("ModelYear")
        make = d.get("Make")
        model = d.get("Model")
        if year or make or model:
            parts.append(f"{year or ''} {make or ''} {model or ''}".strip())
        if d.get("Trim") or d.get("Series"):
            parts.append(f"Trim/Series: {', '.join([x for x in [d.get('Series'), d.get('Trim')] if x])}")
        if d.get("VehicleType") or d.get("BodyClass"):
            parts.append(f"Type: {', '.join([x for x in [d.get('VehicleType'), d.get('BodyClass')] if x])}")
        if d.get("DisplacementL") or d.get("EngineCylinders") or d.get("FuelTypePrimary"):
            eng = []
            if d.get("DisplacementL"):
                eng.append(f"{d['DisplacementL']}L")
            if d.get("EngineCylinders"):
                eng.append(f"{d['EngineCylinders']} cyl")
            if d.get("FuelTypePrimary"):
                eng.append(str(d["FuelTypePrimary"]))
            parts.append(f"Engine: {' / '.join(eng)}")
        if d.get("DriveType"):
            parts.append(f"Drive: {d['DriveType']}")
        if d.get("TransmissionStyle") or d.get("TransmissionSpeeds"):
            tx = []
            if d.get("TransmissionStyle"):
                tx.append(str(d["TransmissionStyle"]))
            if d.get("TransmissionSpeeds"):
                tx.append(f"{d['TransmissionSpeeds']} spd")
            parts.append(f"Transmission: {' / '.join(tx)}")
        if d.get("PlantCountry") or d.get("PlantCity"):
            plant = ", ".join([x for x in [d.get("PlantCity"), d.get("PlantState"), d.get("PlantCountry")] if x])
            parts.append(f"Plant: {plant}")
        return "\n".join(f"- {p}" for p in parts if p)

    if profile.source == "carquery_trims":
        trims = (d.get("trims") or [])[:5]
        out = [f"Top trims found for: {d.get('query', profile.query)}"]
        for t in trims:
            label = " ".join([str(x) for x in [t.get("model_year"), t.get("model_make_id"), t.get("model_name"), t.get("model_trim")] if x]).strip()
            line = f"- {label}" if label else "- Trim"
            extras = []
            if t.get("model_drive"):
                extras.append(str(t["model_drive"]))
            if t.get("model_transmission_type"):
                extras.append(str(t["model_transmission_type"]))
            if t.get("model_engine_fuel"):
                extras.append(str(t["model_engine_fuel"]))
            if t.get("model_engine_cc"):
                extras.append(f"{t['model_engine_cc']} cc")
            if extras:
                line += f" ({', '.join(extras)})"
            out.append(line)
        return "\n".join(out)

    if profile.source == "vpci_models_for_make_year":
        mk = d.get("make") or ""
        y = d.get("year") or ""
        models = d.get("models") or []
        show = ", ".join(models[:30])
        suffix = "" if len(models) <= 30 else f" … (+{len(models) - 30} more)"
        return "\n".join(
            [
                f"Models for {mk.title()} in {y}:",
                f"- {show}{suffix}",
            ]
        )

    if profile.source == "vpci_models_for_make":
        mk = d.get("make") or ""
        models = d.get("models") or []
        show = ", ".join(models[:30])
        suffix = "" if len(models) <= 30 else f" … (+{len(models) - 30} more)"
        return "\n".join(
            [
                f"Models for {mk.title()} (all years):",
                f"- {show}{suffix}",
            ]
        )

    return json.dumps(d, ensure_ascii=False, indent=2)

