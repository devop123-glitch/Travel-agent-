import os
import re
import certifi
import requests
import airportsdata
import pycountry
from dotenv import load_dotenv

load_dotenv()
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
API_KEY = os.getenv("AVIATIONSTACK_API_KEY")
DEFAULT_ORIGIN_IATA = os.getenv("DEAFAULT_ORIGIN_IATA", "CCU")
BASE_URL = "http://api.aviationstack.com/v1/flights"
AIRPORTS = airportsdata.load("IATA")
COUNTRY_ALIASES = {
    "usa": "US",
    "u.s.a": "US",
    "u.s.": "US",
    "america": "US",
    "united states": "US",
    "uk": "GB",
    "u.k.": "GB",
    "britain": "GB",
    "england": "GB",
    "uae": "AE",
    "dubai": "AE",
    "south korea": "KR",
    "korea": "KR",
    "russia": "RU",
    "vietnam": "VN",
    "bangladesh": "BD",
    "india": "IN",
    "japan": "JP",
    "china": "CN",
    "singapore": "SG",
    "malaysia": "MY",
    "thailand": "TH",
    "indonesia": "ID",
    "nepal": "NP",
    "qatar": "QA",
    "saudi arabia": "SA",
    "turkey": "TR",
    "canada": "CA",
    "australia": "AU",
    "germany": "DE",
    "france": "FR",
    "italy": "IT",
    "spain": "ES",
}
COUNTRY_MAIN_AIRPORT = {
    "BD": "DAC",
    "IN": "DEL",
    "JP": "NRT",
    "US": "JFK",
    "GB": "LHR",
    "AE": "DXB",
    "SG": "SIN",
    "MY": "KUL",
    "TH": "BKK",
    "ID": "CGK",
    "CN": "PEK",
    "KR": "ICN",
    "NP": "KTM",
    "QA": "DOH",
    "SA": "JED",
    "TR": "IST",
    "CA": "YYZ",
    "AU": "SYD",
    "DE": "FRA",
    "FR": "CDG",
    "IT": "FCO",
    "ES": "MAD",
}
CITY_MAIN_AIRPORT = {
    "dhaka": "DAC",
    "delhi": "DEL",
    "new delhi": "DEL",
    "mumbai": "BOM",
    "kolkata": "CCU",
    "chennai": "MAA",
    "bangalore": "BLR",
    "bengaluru": "BLR",
    "tokyo": "NRT",
    "osaka": "KIX",
    "kyoto": "KIX",
    "new york": "JFK",
    "london": "LHR",
    "dubai": "DXB",
    "singapore": "SIN",
    "kuala lumpur": "KUL",
    "bangkok": "BKK",
    "doha": "DOH",
    "istanbul": "IST",
    "toronto": "YYZ",
    "sydney": "SYD",
    "paris": "CDG",
    "rome": "FCO",
    "madrid": "MAD",
    "frankfurt": "FRA",
}


def clean_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    stop_words = [
        "flight", "flights", "ticket", "tickets", "trip", "travel",
        "plan", "complete", "days", "day", "including", "hotel",
        "hotels", "sightseeing", "under", "budget", "info", "information"
    ]
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words).strip()


def country_name_to_code(text: str):
    text = clean_text(text)

    if text in COUNTRY_ALIASES:
        return COUNTRY_ALIASES[text]

    try:
        country = pycountry.countries.lookup(text)
        return country.alpha_2
    except LookupError:
        pass

    for country in pycountry.countries:
        country_name = country.name.lower()
        if country_name in text:
            return country.alpha_2

    for alias, code in COUNTRY_ALIASES.items():
        if alias in text:
            return code

    return None


def airport_country_matches(airport: dict, country_code: str) -> bool:
    airport_country = str(airport.get("country", "")).upper().strip()

    if airport_country == country_code:
        return True

    try:
        country = pycountry.countries.get(alpha_2=country_code)
        if country and airport_country.lower() == country.name.lower():
            return True
    except Exception:
        pass

    return False


def get_best_airport_for_country(country_code: str):
    preferred = COUNTRY_MAIN_AIRPORT.get(country_code)

    if preferred and preferred in AIRPORTS:
        return preferred

    candidates = []

    for iata, airport in AIRPORTS.items():
        if not iata:
            continue

        if airport_country_matches(airport, country_code):
            name = str(airport.get("name", "")).lower()
            city = str(airport.get("city", "")).lower()

            score = 0

            if "international" in name:
                score += 50
            if "intl" in name:
                score += 40
            if "capital" in name:
                score += 20
            if city:
                score += 5

            candidates.append((score, iata))

    if not candidates:
        return None

    candidates.sort(reverse=True)
    return candidates[0][1]


def resolve_location_to_iata(location: str):
    """
    Converts country/city/airport/IATA into IATA code.

    Examples:
    Bangladesh -> DAC
    Japan -> NRT
    Dhaka -> DAC
    Tokyo -> NRT
    DAC -> DAC
    """

    if not location:
        return None

    raw_location = location.strip()

    if re.fullmatch(r"[A-Za-z]{3}", raw_location):
        code = raw_location.upper()
        if code in AIRPORTS:
            return code

    location_clean = clean_text(raw_location)

    if not location_clean:
        return None

    if location_clean in CITY_MAIN_AIRPORT:
        return CITY_MAIN_AIRPORT[location_clean]

    country_code = country_name_to_code(location_clean)
    if country_code:
        airport = get_best_airport_for_country(country_code)
        if airport:
            return airport

    city_matches = []

    for iata, airport in AIRPORTS.items():
        city = str(airport.get("city", "")).lower().strip()
        name = str(airport.get("name", "")).lower().strip()

        score = 0

        if city == location_clean:
            score += 100
        elif location_clean in city:
            score += 70

        if location_clean in name:
            score += 50

        if "international" in name:
            score += 10

        if score > 0:
            city_matches.append((score, iata))

    if city_matches:
        city_matches.sort(reverse=True)
        return city_matches[0][1]

    return None


def find_location_mentions(query: str):
    """
    Finds country or city names inside a natural language query.
    """

    q = query.lower()
    mentions = []

    for alias in COUNTRY_ALIASES:
        if re.search(rf"\b{re.escape(alias)}\b", q):
            mentions.append(alias)

    for country in pycountry.countries:
        name = country.name.lower()
        if len(name) >= 4 and re.search(rf"\b{re.escape(name)}\b", q):
            mentions.append(name)

    for city in CITY_MAIN_AIRPORT:
        if re.search(rf"\b{re.escape(city)}\b", q):
            mentions.append(city)

    unique_mentions = []
    for item in mentions:
        if item not in unique_mentions:
            unique_mentions.append(item)

    return unique_mentions


def parse_route(query: str):
    """
    Returns:
    dep_iata, arr_iata

    Can return:
    None, None  -> global live flights
    DAC, NRT    -> filtered route
    DAC, None   -> all flights from DAC
    None, NRT   -> all flights to NRT
    """

    q = query.strip()
    q_lower = q.lower()

    global_keywords = [
        "all country",
        "all countries",
        "global flight",
        "global flights",
        "all flight",
        "all flights",
        "worldwide flight",
        "worldwide flights",
    ]

    if any(keyword in q_lower for keyword in global_keywords):
        return None, None

    codes = re.findall(r"\b[A-Z]{3}\b", q)

    if len(codes) >= 2:
        dep = codes[0].upper()
        arr = codes[1].upper()
        return dep, arr

    match = re.search(
        r"\bfrom\s+(.+?)\s+\bto\s+(.+?)(?:\s+(?:on|for|under|including|with|in|at)\b|[.!?]|$)",
        q_lower,
    )

    if match:
        origin_text = match.group(1)
        dest_text = match.group(2)

        dep_iata = resolve_location_to_iata(origin_text)
        arr_iata = resolve_location_to_iata(dest_text)

        return dep_iata, arr_iata

    match = re.search(
        r"\bto\s+(.+?)\s+\bfrom\s+(.+?)(?:\s+(?:on|for|under|including|with|in|at)\b|[.!?]|$)",
        q_lower,
    )

    if match:
        dest_text = match.group(1)
        origin_text = match.group(2)

        dep_iata = resolve_location_to_iata(origin_text)
        arr_iata = resolve_location_to_iata(dest_text)

        return dep_iata, arr_iata

    match = re.search(r"\bfrom\s+(.+?)(?:[.!?]|$)", q_lower)

    if match:
        origin_text = match.group(1)
        dep_iata = resolve_location_to_iata(origin_text)
        return dep_iata, None

    match = re.search(r"\bto\s+(.+?)(?:[.!?]|$)", q_lower)

    if match:
        dest_text = match.group(1)
        arr_iata = resolve_location_to_iata(dest_text)
        return None, arr_iata

    mentions = find_location_mentions(q)

    if len(mentions) >= 2:
        dep_iata = resolve_location_to_iata(mentions[0])
        arr_iata = resolve_location_to_iata(mentions[1])
        return dep_iata, arr_iata

    if len(mentions) == 1:
        arr_iata = resolve_location_to_iata(mentions[0])
        return DEFAULT_ORIGIN_IATA, arr_iata

    return None, None


def format_flight(flight: dict):
    airline = flight.get("airline", {}).get("name") or "Unknown airline"
    flight_number = flight.get("flight", {}).get("iata") or "Unknown flight number"
    status = flight.get("flight_status") or "Unknown"

    dep = flight.get("departure", {}) or {}
    arr = flight.get("arrival", {}) or {}

    dep_airport = dep.get("airport") or "Unknown departure airport"
    dep_iata = dep.get("iata") or "Unknown"
    dep_terminal = dep.get("terminal") or "N/A"
    dep_gate = dep.get("gate") or "N/A"
    dep_scheduled = dep.get("scheduled") or "Unknown"
    dep_delay = dep.get("delay")
    dep_delay_text = f"{dep_delay} minutes" if dep_delay is not None else "N/A"

    arr_airport = arr.get("airport") or "Unknown arrival airport"
    arr_iata = arr.get("iata") or "Unknown"
    arr_terminal = arr.get("terminal") or "N/A"
    arr_gate = arr.get("gate") or "N/A"
    arr_scheduled = arr.get("scheduled") or "Unknown"
    arr_delay = arr.get("delay")
    arr_delay_text = f"{arr_delay} minutes" if arr_delay is not None else "N/A"

    return f"""
Airline: {airline}
Flight: {flight_number}
Status: {status}

Departure:
- Airport: {dep_airport}
- IATA: {dep_iata}
- Terminal: {dep_terminal}
- Gate: {dep_gate}
- Scheduled: {dep_scheduled}
- Delay: {dep_delay_text}

Arrival:
- Airport: {arr_airport}
- IATA: {arr_iata}
- Terminal: {arr_terminal}
- Gate: {arr_gate}
- Scheduled: {arr_scheduled}
- Delay: {arr_delay_text}
""".strip()


def _fetch_flights_raw(query: str, limit: int = 10):
    """
    Internal helper shared by search_flights() and get_raw_flights().
    Talks to AviationStack once and returns a plain dict so both callers
    can decide how to present the result - nothing here raises.

    Returns:
    {
        "flights": [...] ,   # list, possibly empty
        "dep_iata": str|None,
        "arr_iata": str|None,
        "error": str|None,   # set only on failure
    }
    """
    result = {"flights": [], "dep_iata": None, "arr_iata": None, "error": None}

    if not API_KEY:
        result["error"] = (
            "Flight API error: AVIATIONSTACK_API_KEY is missing.\n"
            "Please add this in your .env file:\n"
            "AVIATIONSTACK_API_KEY=your_api_key_here"
        )
        return result

    try:
        dep_iata, arr_iata = parse_route(query)
    except Exception as e:
        result["error"] = f"Could not parse route from query: {e}"
        return result

    result["dep_iata"] = dep_iata
    result["arr_iata"] = arr_iata

    params = {
        "access_key": API_KEY,
        "limit": min(limit, 100),
    }

    if dep_iata:
        params["dep_iata"] = dep_iata
    if arr_iata:
        params["arr_iata"] = arr_iata

    try:
        response = requests.get(BASE_URL, params=params, timeout=30)
    except requests.exceptions.RequestException as e:
        result["error"] = f"Flight API request failed: {e}"
        return result

    try:
        data = response.json()
    except ValueError:
        result["error"] = "Flight API returned invalid JSON."
        return result

    if not isinstance(data, dict):
        result["error"] = "Flight API returned an unexpected response format."
        return result

    if "error" in data:
        error = data["error"]
        if isinstance(error, dict):
            result["error"] = (
                "Flight API error:\n"
                f"Code: {error.get('code', 'Unknown')}\n"
                f"Message: {error.get('message', 'Unknown error')}"
            )
        else:
            result["error"] = f"Flight API error: {error}"
        return result

    flights = data.get("data", [])
    result["flights"] = flights if isinstance(flights, list) else []

    return result


def search_flights(query: str, limit: int = 10):
    fetched = _fetch_flights_raw(query, limit)

    if fetched["error"]:
        return fetched["error"]

    dep_iata = fetched["dep_iata"]
    arr_iata = fetched["arr_iata"]
    flight_data = fetched["flights"]

    if not flight_data:
        route_text = ""

        if dep_iata and arr_iata:
            route_text = f" for route {dep_iata} to {arr_iata}"
        elif dep_iata:
            route_text = f" from {dep_iata}"
        elif arr_iata:
            route_text = f" to {arr_iata}"

        return (
            f"No live flight data found{route_text}.\n\n"
            "Note: AviationStack provides live/status flight data, not ticket prices. "
            "For actual fare prices, use a flight-pricing API such as Amadeus."
        )

    route_info = "Global live flights"

    if dep_iata and arr_iata:
        route_info = f"Live flights from {dep_iata} to {arr_iata}"
    elif dep_iata:
        route_info = f"Live flights from {dep_iata}"
    elif arr_iata:
        route_info = f"Live flights to {arr_iata}"

    formatted_flights = [format_flight(flight) for flight in flight_data[:limit]]

    return f"{route_info}\n\n" + "\n\n---\n\n".join(formatted_flights)


def get_raw_flights(query: str, limit: int = 10):
    """
    Same lookup as search_flights(), but returns the raw AviationStack
    flight list (list of dicts) instead of formatted text - or an error
    string if the request failed / no API key is configured.

    Use this (plus extract_airports_and_airlines below) when you need
    structured airport/airline data instead of human-readable text, e.g.
    to feed an LLM prompt, without relying on separate list_airports /
    list_airlines endpoints that may be restricted on your plan.
    """
    fetched = _fetch_flights_raw(query, limit)

    if fetched["error"]:
        return fetched["error"]

    return fetched["flights"]


def extract_airports_and_airlines(flights):
    """
    Pulls unique airports and airlines out of a raw AviationStack flight
    list (as returned by get_raw_flights). This replaces separate
    'list_airports' / 'list_airlines' lookups - the same flight records
    already contain this info.

    Accepts:
      - a list of flight dicts (normal case)
      - an error string from get_raw_flights (handled gracefully)
      - None / anything else (handled gracefully)

    Returns (airports, airlines):
      airports -> [{"iata": "DAC", "airport": "...", "timezone": "..."}, ...]
      airlines -> [{"iata": "BG", "name": "Biman Bangladesh"}, ...]

    Never raises - always returns ([], []) on malformed input.
    """
    if not isinstance(flights, list):
        return [], []

    seen_airports = {}
    seen_airlines = {}

    for flight in flights:
        if not isinstance(flight, dict):
            continue

        dep = flight.get("departure") or {}
        arr = flight.get("arrival") or {}
        airline = flight.get("airline") or {}

        if not isinstance(dep, dict):
            dep = {}
        if not isinstance(arr, dict):
            arr = {}
        if not isinstance(airline, dict):
            airline = {}

        dep_iata = dep.get("iata")
        if dep_iata and dep_iata not in seen_airports:
            seen_airports[dep_iata] = {
                "iata": dep_iata,
                "airport": dep.get("airport") or "Unknown",
                "timezone": dep.get("timezone"),
            }

        arr_iata = arr.get("iata")
        if arr_iata and arr_iata not in seen_airports:
            seen_airports[arr_iata] = {
                "iata": arr_iata,
                "airport": arr.get("airport") or "Unknown",
                "timezone": arr.get("timezone"),
            }

        key = airline.get("iata") or airline.get("name")
        if key and key not in seen_airlines:
            seen_airlines[key] = {
                "iata": airline.get("iata"),
                "name": airline.get("name") or "Unknown airline",
            }

    return list(seen_airports.values()), list(seen_airlines.values())


if __name__ == "__main__":
    print(search_flights("Plan a 7 days Japan trip from Kolkata including flights and hotels"))
    print("\n" + "=" * 80 + "\n")
    print(search_flights("all country flight info"))

    print("\n" + "=" * 80 + "\n")
    raw = get_raw_flights("all country flight info")
    airports, airlines = extract_airports_and_airlines(raw)
    print("Airports:", airports)
    print("Airlines:", airlines)