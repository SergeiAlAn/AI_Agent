"""CLI-интерфейс агента.

Примеры:
    python run.py                         # интерактивный режим
    python run.py -v                      # + подробный лог в консоли
    python run.py -q "Сколько стоит Bitcoin?"   # один вопрос и выход
"""

from __future__ import annotations

import argparse
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
LOG_FILE = BASE_DIR / "logs" / "agent.log"

HELP = """Команды:
  /help    — эта справка
  /memory  — показать резюме из долговременной памяти
  /clear   — очистить память (memory.json)
  /tools   — список инструментов
  /exit    — выход (также: exit, quit, Ctrl+C)"""


class ConsoleFormatter(logging.Formatter):
    """Без трейсбеков: в консоли только суть ошибки, полный стек — в logs/agent.log."""

    def format(self, record: logging.LogRecord) -> str:
        record = logging.makeLogRecord(record.__dict__)
        record.exc_info = record.exc_text = None
        return super().format(record)


def setup_logging(verbose: bool) -> None:
    LOG_FILE.parent.mkdir(exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")

    file_handler = RotatingFileHandler(LOG_FILE, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.INFO if verbose else logging.WARNING)
    console.setFormatter(ConsoleFormatter("  [%(levelname)s] %(name)s: %(message)s"))

    root = logging.getLogger()
    root.setLevel(logging.WARNING)
    root.addHandler(file_handler)
    root.addHandler(console)
    logging.getLogger("agent").setLevel(logging.DEBUG)
    for noisy in ("httpx", "httpcore", "openai", "urllib3", "primp", "ddgs"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Локальный AI-агент через ProxyAPI")
    parser.add_argument("-q", "--query", help="задать один вопрос и выйти")
    parser.add_argument("-v", "--verbose", action="store_true", help="показывать шаги агента в консоли")
    args = parser.parse_args()

    setup_logging(args.verbose)
    log = logging.getLogger("agent.cli")

    from agent import Agent, Memory  # импорт после настройки логов

    try:
        agent = Agent(Memory())
    except Exception as exc:
        log.exception("Не удалось инициализировать агента")
        print(f"Ошибка запуска: {exc}", file=sys.stderr)
        return 1

    def handle(question: str) -> None:
        try:
            result = agent.ask(question)
        except Exception as exc:
            log.exception("Ошибка при обработке запроса")
            print(f"\nОшибка: {type(exc).__name__}: {exc}\nПодробности: {LOG_FILE}\n")
            return
        tools = f" | инструменты: {', '.join(result['tools'])}" if result["tools"] else ""
        print(f"\nАгент:\n{result['answer']}\n")
        print(f"  ({result['elapsed']:.1f} c{tools})\n")

    if args.query:
        handle(args.query)
        return 0

    print("AI-агент запущен. Задайте вопрос на естественном языке. /help — команды.\n")
    while True:
        try:
            question = input("Вы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nДо встречи!")
            return 0
        if not question:
            continue
        cmd = question.lower()
        if cmd in ("/exit", "exit", "quit", "выход"):
            print("До встречи!")
            return 0
        if cmd == "/help":
            print(HELP + "\n")
        elif cmd == "/memory":
            print("\n" + (agent.memory.summary or "Память пуста.") + "\n")
        elif cmd == "/clear":
            agent.memory.clear()
            print("Память очищена.\n")
        elif cmd == "/tools":
            for t in agent.tools:
                print(f"  - {t.name}: {t.description.splitlines()[0]}")
            print()
        else:
            handle(question)


if __name__ == "__main__":
    sys.exit(main())
