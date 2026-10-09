"""Быстрая проверка инструментов без LLM: python test_tools.py"""

import logging
import sys

sys.stdout.reconfigure(encoding="utf-8")
logging.basicConfig(level=logging.WARNING)

from tools import ALL_TOOLS, get_crypto_price, get_currency_cb_rate, get_share_price  # noqa: E402

tools = {t.name: t for t in ALL_TOOLS}

checks = [
    ("get_weather", {"city": "Москва"}),
    ("get_crypto_price", {"coin": "bitcoin", "currency": "usd"}),
    ("get_crypto_price", {"coin": "eth", "currency": "rub"}),
    ("get_currency_cb_rate", {"currency_code": "USD", "on_date": "02.10.2026"}),
    ("get_currency_cb_rate", {"currency_code": "AMD"}),
    ("get_share_price", {"ticker": "SBER"}),
    ("get_share_price", {"ticker": "Газпром"}),
    ("web_search", {"query": "Python 3.14 release", "max_results": 2}),
    ("http_request", {"url": "https://httpbin.org/get", "params": {"a": "1"}}),
    ("http_request", {"url": "http://localhost:8000"}),
    ("write_file", {"path": "test/hello.txt", "content": "Привет"}),
    ("read_file", {"path": "test/hello.txt"}),
    ("read_file", {"path": "../.env"}),
    ("list_directory", {"path": "."}),
    ("run_terminal_command", {"command": "dir"}),
    ("run_terminal_command", {"command": "echo hi & del x"}),
    ("run_terminal_command", {"command": "rm -rf /"}),
    ("run_terminal_command", {"command": "python -c print(1)"}),
    ("run_terminal_command", {"command": "type ..\\.env"}),
    ("run_terminal_command", {"command": "dir C:\\"}),
    ("run_terminal_command", {"command": "dir /b"}),
]

for name, args in checks:
    print(f"=== {name} {args}")
    print(tools[name].invoke(args))
    print()

print("float API:", get_crypto_price("bitcoin", "usd"), get_currency_cb_rate("EUR"), get_share_price("LKOH"))
