"""Инструменты агента.

Каждая "бизнес"-функция (get_weather, get_crypto_price, ...) — обычная Python-функция,
которую можно вызывать и тестировать отдельно. Ниже они обёрнуты в LangChain-инструменты
(@tool), которые возвращают агенту текст и никогда не роняют цикл агента исключением.
"""

from __future__ import annotations

import functools
import ipaddress
import json
import logging
import os
import platform
import re
import shlex
import socket
import subprocess
import threading
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import urlparse

import requests
from langchain_core.tools import tool

log = logging.getLogger("agent.tools")

BASE_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = Path(os.getenv("AGENT_WORKSPACE", BASE_DIR / "workspace")).resolve()
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

HTTP_TIMEOUT = 20
MAX_TOOL_OUTPUT = 6000
HEADERS = {"User-Agent": "Mozilla/5.0 (local-ai-agent)"}
BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"
    ),
    "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
}


def _truncate(text: str, limit: int = MAX_TOOL_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [обрезано, всего {len(text)} символов]"


def logged_tool(func: Callable[..., str]) -> Callable[..., str]:
    """Логирует аргументы, время и результат вызова; превращает исключения в текст ошибки."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> str:
        started = time.perf_counter()
        log.info("TOOL CALL %s args=%s kwargs=%s", func.__name__, args, kwargs)
        try:
            result = func(*args, **kwargs)
        except (PermissionError, ValueError) as exc:
            log.warning("TOOL REJECTED %s: %s", func.__name__, exc)
            return f"Ошибка инструмента {func.__name__}: {type(exc).__name__}: {exc}"
        except Exception as exc:
            elapsed = time.perf_counter() - started
            log.exception("TOOL FAIL %s (%.2fs): %s", func.__name__, elapsed, exc)
            return f"Ошибка инструмента {func.__name__}: {type(exc).__name__}: {exc}"
        elapsed = time.perf_counter() - started
        result = _truncate(str(result))
        log.info("TOOL OK   %s (%.2fs) -> %s", func.__name__, elapsed, result[:300].replace("\n", " "))
        return result

    return wrapper


# ---------------------------------------------------------------------------
# 1. Web Search (DuckDuckGo)
# ---------------------------------------------------------------------------

try:
    from ddgs import DDGS  # новое имя пакета
except ImportError:  # pragma: no cover
    from duckduckgo_search import DDGS  # старое имя пакета duckduckgo-search

# Пары backend:region пробуются по очереди, пока какая-то не вернёт результаты.
# Сам DuckDuckGo из РФ часто отвечает пусто; yahoo быстрый (~2 c), auto медленный (~10 c), но
# перебирает несколько движков. Доступные бэкенды: auto, yahoo, duckduckgo, google, brave,
# mojeek, startpage, wikipedia.
SEARCH_ATTEMPTS = [
    tuple(item.strip().split(":", 1)) if ":" in item else (item.strip(), "wt-wt")
    for item in os.getenv(
        "AGENT_SEARCH_BACKENDS", "yahoo:wt-wt,yahoo:ru-ru,duckduckgo:ru-ru,auto:ru-ru"
    ).split(",")
    if item.strip()
]
SEARCH_TIMEOUT = 10
SEARCH_DEADLINE = 45
# Параллельные запросы от агента поисковики режут по rate limit — выполняем их по одному.
_search_lock = threading.Lock()


def _simplify_query(query: str) -> str:
    simplified = re.sub(r"\bsite:\S+", " ", query)
    simplified = re.sub(r'["«»]', " ", simplified)
    return " ".join(simplified.split()[:8])


def search_web(query: str, max_results: int = 5) -> list[dict]:
    """Ищет с перебором бэкендов и упрощением запроса; пустой список = ничего не найдено."""
    attempts = [(query, b, r) for b, r in SEARCH_ATTEMPTS]
    simple = _simplify_query(query)
    if simple and simple != query:
        attempts += [(simple, b, r) for b, r in SEARCH_ATTEMPTS]

    started = time.monotonic()
    with _search_lock:
        for q, backend, region in attempts:
            if time.monotonic() - started > SEARCH_DEADLINE:
                log.warning("Поиск: превышен общий лимит %ss для '%s'", SEARCH_DEADLINE, query)
                break
            try:
                results = DDGS(timeout=SEARCH_TIMEOUT).text(
                    q, region=region, max_results=max_results, backend=backend
                )
            except Exception as exc:
                log.info("Поиск: %s/%s query='%s' -> %s: %s", backend, region, q, type(exc).__name__, exc)
                continue
            if results:
                log.info("Поиск: %s/%s query='%s' -> %d результатов", backend, region, q, len(results))
                return [
                    {"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")}
                    for r in results
                ]
    return []


@tool
@logged_tool
def web_search(query: str, max_results: int = 5) -> str:
    """Поиск актуальной информации в интернете (DuckDuckGo/Yahoo через ddgs).
    Используй для новостей, фактов, описаний, всего, чего нет в других инструментах.
    Запрос делай коротким (3-7 слов). Возвращает заголовок, ссылку и краткий фрагмент;
    чтобы прочитать саму страницу, вызови fetch_webpage."""
    results = search_web(query, max_results=max(1, min(max_results, 10)))
    if not results:
        return (
            f"По запросу '{query}' ничего не найдено. Попробуй переформулировать: "
            "короче, без кавычек и оператора site:, другими словами или на английском."
        )
    return "\n\n".join(
        f"{i}. {r['title']}\n   {r['url']}\n   {r['snippet']}" for i, r in enumerate(results, 1)
    )


class _TextExtractor(HTMLParser):
    SKIP = {"script", "style", "noscript", "svg", "head", "template"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.title = ""
        self._skip_depth = 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip_depth += 1
        if tag == "title":
            self._in_title = True

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip_depth:
            self._skip_depth -= 1
        if tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self._skip_depth and data.strip():
            self.parts.append(data.strip())


@tool
@logged_tool
def fetch_webpage(url: str) -> str:
    """Открыть веб-страницу и вернуть её текст без HTML-разметки (например, страницу из результатов web_search).
    Сайты, которые строят содержимое JavaScript'ом (интернет-магазины, SPA), могут вернуть почти пустой текст."""
    _ensure_public_url(url)
    resp = requests.get(url, headers=BROWSER_HEADERS, timeout=HTTP_TIMEOUT)
    resp.raise_for_status()
    if "html" not in resp.headers.get("Content-Type", "html"):
        return resp.text
    parser = _TextExtractor()
    parser.feed(resp.text)
    text = "\n".join(parser.parts)
    if len(text) < 200:
        text += (
            "\n\n[На странице почти нет текста: вероятно, содержимое загружается JavaScript'ом "
            "или сайт блокирует автоматические запросы. Используй другие источники.]"
        )
    return f"Страница: {parser.title.strip()}\nURL: {resp.url}\n\n{text}"


# ---------------------------------------------------------------------------
# 2. HTTP API Requests
# ---------------------------------------------------------------------------

def _ensure_public_url(url: str) -> None:
    """Защита от запросов во внутреннюю сеть (localhost, 192.168.x.x и т.п.)."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("Разрешены только абсолютные http/https URL")
    for info in socket.getaddrinfo(parsed.hostname, None):
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            raise ValueError(f"Запросы к внутренним адресам запрещены ({ip})")


@tool
@logged_tool
def http_request(
    url: str,
    method: str = "GET",
    params: Optional[dict] = None,
    json_body: Optional[dict] = None,
    headers: Optional[dict] = None,
) -> str:
    """Выполнить HTTP-запрос к внешнему публичному API.
    method: GET, POST или HEAD. params — query-параметры, json_body — тело запроса (для POST).
    Возвращает статус, content-type и тело ответа (обрезанное)."""
    method = method.upper()
    if method not in ("GET", "POST", "HEAD"):
        raise ValueError("Разрешены только методы GET, POST, HEAD")
    _ensure_public_url(url)
    resp = requests.request(
        method,
        url,
        params=params,
        json=json_body,
        headers={**HEADERS, **(headers or {})},
        timeout=HTTP_TIMEOUT,
    )
    content_type = resp.headers.get("Content-Type", "")
    body = resp.text
    if "json" in content_type:
        try:
            body = json.dumps(resp.json(), ensure_ascii=False, indent=2)
        except ValueError:
            pass
    return f"HTTP {resp.status_code}\nContent-Type: {content_type}\n\n{body}"


# ---------------------------------------------------------------------------
# 3. Файловая система (только внутри папки workspace/)
# ---------------------------------------------------------------------------

def _safe_path(path: str) -> Path:
    candidate = (WORKSPACE_DIR / path).resolve()
    if not candidate.is_relative_to(WORKSPACE_DIR):
        raise PermissionError(f"Доступ разрешён только внутри {WORKSPACE_DIR}")
    return candidate


@tool
@logged_tool
def read_file(path: str) -> str:
    """Прочитать текстовый файл из рабочей папки агента (workspace). path — относительный путь."""
    target = _safe_path(path)
    if not target.is_file():
        return f"Файл не найден: {path}"
    return target.read_text(encoding="utf-8", errors="replace")


@tool
@logged_tool
def write_file(path: str, content: str, append: bool = False) -> str:
    """Записать текст в файл в рабочей папке агента (workspace).
    append=True — дописать в конец, иначе файл перезаписывается. Папки создаются автоматически."""
    target = _safe_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a" if append else "w", encoding="utf-8") as fh:
        fh.write(content)
    return f"Записано {len(content)} символов в {target.relative_to(WORKSPACE_DIR)}"


@tool
@logged_tool
def list_directory(path: str = ".") -> str:
    """Показать содержимое папки внутри рабочей папки агента (workspace)."""
    target = _safe_path(path)
    if not target.is_dir():
        return f"Папка не найдена: {path}"
    lines = []
    for item in sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name.lower())):
        if item.is_dir():
            lines.append(f"[DIR]  {item.name}/")
        else:
            lines.append(f"[FILE] {item.name} ({item.stat().st_size} байт)")
    return "\n".join(lines) or "Папка пуста."


# ---------------------------------------------------------------------------
# 4. Терминальные команды (белый список, без shell-операторов)
# ---------------------------------------------------------------------------

IS_WINDOWS = platform.system() == "Windows"

# команда -> разрешённые подкоманды/первые аргументы (None = любые аргументы)
ALLOWED_COMMANDS: dict[str, Optional[set[str]]] = {
    "dir": None,
    "ls": None,
    "tree": None,
    "type": None,
    "cat": None,
    "echo": None,
    "whoami": None,
    "hostname": None,
    "ver": None,
    "systeminfo": None,
    "ipconfig": None,
    "ping": None,
    "where": None,
    "findstr": None,
    "python": {"--version", "-V"},
    "py": {"--version", "-V"},
    "pip": {"list", "show", "freeze", "--version"},
    "git": {"status", "log", "diff", "branch", "--version"},
}
FORBIDDEN_CHARS = set("&|;<>`$()\n\r%^")
COMMAND_TIMEOUT = 30


def _escapes_workspace(arg: str) -> bool:
    if ".." in arg or arg.startswith(("~", "\\\\")) or re.match(r"^[a-zA-Z]:", arg):
        return True
    # в Windows аргументы вида /s, /b — это флаги, а не пути
    return not IS_WINDOWS and arg.startswith("/")


def _console_encoding() -> str:
    if IS_WINDOWS:
        import ctypes

        return f"cp{ctypes.windll.kernel32.GetOEMCP()}"
    return "utf-8"


@tool
@logged_tool
def run_terminal_command(command: str) -> str:
    """Безопасно выполнить терминальную команду из белого списка в рабочей папке workspace.
    Разрешены: dir, ls, tree, type, cat, echo, whoami, hostname, ver, systeminfo, ipconfig,
    ping, where, findstr, python/py --version, pip list|show|freeze, git status|log|diff|branch.
    Операторы &, |, ;, >, <, $, % и т.п. запрещены. Таймаут 30 секунд."""
    command = command.strip()
    if not command:
        raise ValueError("Пустая команда")
    bad = FORBIDDEN_CHARS.intersection(command)
    if bad:
        raise PermissionError(f"Запрещённые символы в команде: {''.join(sorted(bad))}")

    parts = shlex.split(command, posix=not IS_WINDOWS)
    name = Path(parts[0]).name.lower().removesuffix(".exe")
    if name not in ALLOWED_COMMANDS:
        raise PermissionError(f"Команда '{name}' не входит в белый список: {', '.join(ALLOWED_COMMANDS)}")
    allowed_sub = ALLOWED_COMMANDS[name]
    if allowed_sub is not None and (len(parts) < 2 or parts[1].lower() not in allowed_sub):
        raise PermissionError(f"Для '{name}' разрешены только: {', '.join(sorted(allowed_sub))}")
    for arg in parts[1:]:
        if _escapes_workspace(arg):
            raise PermissionError(f"Аргумент '{arg}' указывает за пределы рабочей папки")
    if name == "ping" and not any(p.lower() in ("-n", "-c") for p in parts):
        parts[1:1] = ["-n" if IS_WINDOWS else "-c", "4"]

    log.debug("Executing: %s (cwd=%s)", parts, WORKSPACE_DIR)
    proc = subprocess.run(
        subprocess.list2cmdline(parts) if IS_WINDOWS else parts,
        shell=IS_WINDOWS,  # встроенные команды cmd (dir, type, echo) требуют оболочки
        cwd=WORKSPACE_DIR,
        capture_output=True,
        timeout=COMMAND_TIMEOUT,
        encoding=_console_encoding(),
        errors="replace",
    )
    output = (proc.stdout or "") + (("\n[stderr]\n" + proc.stderr) if proc.stderr else "")
    return f"exit code: {proc.returncode}\n{output.strip() or '(нет вывода)'}"


# ---------------------------------------------------------------------------
# 5. Погода (Open-Meteo + геокодинг)
# ---------------------------------------------------------------------------

WMO_CODES = {
    0: "ясно", 1: "преимущественно ясно", 2: "переменная облачность", 3: "пасмурно",
    45: "туман", 48: "изморозь, туман", 51: "лёгкая морось", 53: "морось", 55: "сильная морось",
    56: "ледяная морось", 57: "сильная ледяная морось", 61: "небольшой дождь", 63: "дождь",
    65: "сильный дождь", 66: "ледяной дождь", 67: "сильный ледяной дождь", 71: "небольшой снег",
    73: "снег", 75: "сильный снег", 77: "снежные зёрна", 80: "небольшой ливень", 81: "ливень",
    82: "сильный ливень", 85: "снегопад", 86: "сильный снегопад", 95: "гроза",
    96: "гроза с небольшим градом", 99: "гроза с сильным градом",
}


def geocode_city(city: str) -> dict:
    resp = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1, "language": "ru", "format": "json"},
        timeout=HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    results = resp.json().get("results")
    if not results:
        raise ValueError(f"Город '{city}' не найден")
    r = results[0]
    return {
        "name": r.get("name"),
        "country": r.get("country"),
        "region": r.get("admin1"),
        "latitude": r["latitude"],
        "longitude": r["longitude"],
    }


def get_weather(city: str) -> dict:
    place = geocode_city(city)
    resp = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "current_weather": "true",
            "timezone": "auto",
        },
        timeout=HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    current = resp.json()["current_weather"]
    return {
        **place,
        "time": current.get("time"),
        "temperature_c": current.get("temperature"),
        "windspeed_kmh": current.get("windspeed"),
        "winddirection_deg": current.get("winddirection"),
        "conditions": WMO_CODES.get(current.get("weathercode"), f"код {current.get('weathercode')}"),
        "is_day": bool(current.get("is_day")),
    }


@tool("get_weather")
@logged_tool
def weather_tool(city: str) -> str:
    """Текущая погода в городе: температура, ветер, условия. city — название города (на любом языке)."""
    w = get_weather(city)
    location = ", ".join(x for x in (w["name"], w["region"], w["country"]) if x)
    return (
        f"Погода: {location} (на {w['time']} местного времени)\n"
        f"Температура: {w['temperature_c']} °C\n"
        f"Ветер: {w['windspeed_kmh']} км/ч, направление {w['winddirection_deg']}°\n"
        f"Условия: {w['conditions']}"
    )


# ---------------------------------------------------------------------------
# 6. Курс криптовалюты (CoinGecko)
# ---------------------------------------------------------------------------

COIN_ALIASES = {
    "btc": "bitcoin", "биткоин": "bitcoin", "биткойн": "bitcoin",
    "eth": "ethereum", "эфир": "ethereum", "эфириум": "ethereum",
    "usdt": "tether", "sol": "solana", "ton": "the-open-network", "тон": "the-open-network",
    "bnb": "binancecoin", "xrp": "ripple", "doge": "dogecoin", "ada": "cardano",
    "trx": "tron", "ltc": "litecoin", "dot": "polkadot",
}


@functools.lru_cache(maxsize=256)
def _resolve_coin_id(coin: str) -> str:
    key = coin.strip().lower()
    if key in COIN_ALIASES:
        return COIN_ALIASES[key]
    resp = requests.get(
        "https://api.coingecko.com/api/v3/search", params={"query": key}, timeout=HTTP_TIMEOUT
    )
    if resp.ok:
        coins = resp.json().get("coins") or []
        for c in coins:
            if key in (c.get("id", "").lower(), c.get("symbol", "").lower(), c.get("name", "").lower()):
                return c["id"]
        if coins:
            return coins[0]["id"]
    return key


def get_crypto_price(coin: str, currency: str = "usd") -> float:
    coin_id = _resolve_coin_id(coin)
    currency = currency.strip().lower()
    resp = requests.get(
        "https://api.coingecko.com/api/v3/simple/price",
        params={"ids": coin_id, "vs_currencies": currency},
        headers=HEADERS,
        timeout=HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    if coin_id not in data or currency not in data[coin_id]:
        raise ValueError(f"CoinGecko не вернул цену для coin='{coin_id}', currency='{currency}'")
    return float(data[coin_id][currency])


@tool("get_crypto_price")
@logged_tool
def crypto_price_tool(coin: str, currency: str = "usd") -> str:
    """Текущая цена криптовалюты через CoinGecko.
    coin — id или тикер монеты (bitcoin, ethereum, btc, eth, solana...),
    currency — валюта котировки (usd, eur, rub...)."""
    price = get_crypto_price(coin, currency)
    formatted = f"{price:,.2f}" if price >= 1 else f"{price:.8g}"
    return f"{_resolve_coin_id(coin)}: {formatted} {currency.upper()} (источник: CoinGecko)"


# ---------------------------------------------------------------------------
# 7. Курс валют ЦБ РФ на дату
# ---------------------------------------------------------------------------

def _parse_date(value: Optional[str]) -> date:
    if not value:
        return date.today()
    for fmt in ("%d.%m.%Y", "%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(value.strip(), fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Не удалось разобрать дату '{value}'. Используйте формат ДД.ММ.ГГГГ")


def fetch_cb_rate(currency_code: str = "USD", on_date: Optional[str] = None) -> dict:
    day = _parse_date(on_date)
    resp = requests.get(
        "https://cbr.ru/scripts/XML_daily.asp",
        params={"date_req": day.strftime("%d/%m/%Y")},
        headers=HEADERS,
        timeout=HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    root = ET.fromstring(resp.content)  # XML в windows-1251, кодировку берёт из заголовка
    code = currency_code.strip().upper()
    for valute in root.findall("Valute"):
        if valute.findtext("CharCode") == code:
            nominal = int(valute.findtext("Nominal"))
            value = float(valute.findtext("Value").replace(",", "."))
            return {
                "code": code,
                "name": valute.findtext("Name"),
                "nominal": nominal,
                "value": value,
                "rate_per_unit": value / nominal,
                "date": root.get("Date"),
                "requested_date": day.strftime("%d.%m.%Y"),
            }
    raise ValueError(f"Валюта '{code}' не найдена в данных ЦБ РФ на {day:%d.%m.%Y}")


def get_currency_cb_rate(currency_code: str = "USD", on_date: Optional[str] = None) -> float:
    """Официальный курс ЦБ РФ: сколько рублей стоит 1 единица валюты."""
    return fetch_cb_rate(currency_code, on_date)["rate_per_unit"]


@tool("get_currency_cb_rate")
@logged_tool
def cb_rate_tool(currency_code: str = "USD", on_date: Optional[str] = None) -> str:
    """Официальный курс валюты ЦБ РФ к рублю на дату.
    currency_code — буквенный код (USD, EUR, CNY, GBP, ...),
    on_date — дата в формате ДД.ММ.ГГГГ (если не указана — сегодня)."""
    r = fetch_cb_rate(currency_code, on_date)
    note = ""
    if r["date"] != r["requested_date"]:
        note = f" (на {r['requested_date']} ЦБ использует курс, установленный на {r['date']})"
    nominal = f"{r['nominal']} {r['code']} = {r['value']} руб., " if r["nominal"] != 1 else ""
    return (
        f"Курс ЦБ РФ: {r['name']} ({r['code']}) на {r['date']}{note}: "
        f"{nominal}1 {r['code']} = {r['rate_per_unit']:.4f} руб."
    )


# ---------------------------------------------------------------------------
# 8. Цена акции на Московской бирже (MOEX ISS)
# ---------------------------------------------------------------------------

MOEX_ISS = "https://iss.moex.com/iss"


def _moex_find_ticker(query: str) -> str:
    """Поиск тикера акции по названию компании (например, 'Сбербанк' -> SBER)."""
    resp = requests.get(
        f"{MOEX_ISS}/securities.json",
        params={
            "q": query,
            "engine": "stock",
            "market": "shares",
            "is_trading": 1,
            "iss.meta": "off",
            "securities.columns": "secid,shortname,primary_boardid",
        },
        timeout=HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    rows = resp.json()["securities"]["data"]
    for secid, _name, board in rows:
        if board == "TQBR":
            return secid
    if rows:
        return rows[0][0]
    raise ValueError(f"Акция '{query}' не найдена на Мосбирже")


def fetch_share_quote(ticker: str, board: str = "TQBR") -> dict:
    def load(secid: str) -> tuple[list, list]:
        resp = requests.get(
            f"{MOEX_ISS}/engines/stock/markets/shares/boards/{board}/securities/{secid}.json",
            params={"iss.meta": "off", "iss.only": "securities,marketdata"},
            timeout=HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        sec = [dict(zip(data["securities"]["columns"], row)) for row in data["securities"]["data"]]
        md = [dict(zip(data["marketdata"]["columns"], row)) for row in data["marketdata"]["data"]]
        return sec, md

    secid = ticker.strip().upper()
    sec, md = load(secid)
    if not sec:
        secid = _moex_find_ticker(ticker)
        sec, md = load(secid)
    if not sec:
        raise ValueError(f"Нет данных по '{ticker}' в режиме торгов {board}")

    s, m = sec[0], (md[0] if md else {})
    for field in ("LAST", "LCURRENTPRICE", "MARKETPRICE", "LEGALCLOSEPRICE"):
        if m.get(field):
            price, source = float(m[field]), field
            break
    else:
        if not s.get("PREVPRICE"):
            raise ValueError(f"Мосбиржа не вернула цену для {secid}")
        price, source = float(s["PREVPRICE"]), "PREVPRICE"

    return {
        "ticker": secid,
        "name": s.get("SHORTNAME"),
        "price": price,
        "price_field": source,
        "currency": "RUB" if s.get("CURRENCYID") in ("SUR", "RUB") else s.get("CURRENCYID"),
        "change_pct": m.get("LASTTOPREVPRICE"),
        "update_time": m.get("UPDATETIME"),
        "board": board,
    }


def get_share_price(ticker: str, board: str = "TQBR") -> float:
    return fetch_share_quote(ticker, board)["price"]


@tool("get_share_price")
@logged_tool
def share_price_tool(ticker: str, board: str = "TQBR") -> str:
    """Текущая цена акции на Московской бирже (MOEX).
    ticker — тикер (SBER, GAZP, LKOH, YDEX...) или название компании ('Сбербанк').
    board — режим торгов, по умолчанию TQBR (основной режим для акций)."""
    q = fetch_share_quote(ticker, board)
    change = f", изменение за день: {q['change_pct']}%" if q["change_pct"] is not None else ""
    field_hint = {
        "LAST": "последняя сделка",
        "LCURRENTPRICE": "текущая цена",
        "MARKETPRICE": "рыночная цена",
        "LEGALCLOSEPRICE": "цена закрытия",
        "PREVPRICE": "цена закрытия предыдущего дня",
    }[q["price_field"]]
    return (
        f"{q['name']} ({q['ticker']}, {q['board']}): {q['price']} {q['currency']} "
        f"[{field_hint}, время обновления {q['update_time']}]{change}"
    )


ALL_TOOLS = [
    web_search,
    fetch_webpage,
    http_request,
    read_file,
    write_file,
    list_directory,
    run_terminal_command,
    weather_tool,
    crypto_price_tool,
    cb_rate_tool,
    share_price_tool,
]
