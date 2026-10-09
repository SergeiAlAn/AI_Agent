"""Логика агента: LLM через ProxyAPI + LangChain tool-calling агент + память в memory.json."""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_openai import ChatOpenAI

from tools import ALL_TOOLS, WORKSPACE_DIR

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

log = logging.getLogger("agent.core")

DEFAULT_BASE_URL = "https://api.proxyapi.ru/openai/v1"
DEFAULT_MODEL = "openai/gpt-5-mini"
MEMORY_FILE = Path(os.getenv("AGENT_MEMORY_FILE", BASE_DIR / "memory.json"))
CONTEXT_MESSAGES = int(os.getenv("AGENT_CONTEXT_MESSAGES", "12"))
MAX_STORED_MESSAGES = int(os.getenv("AGENT_MAX_STORED_MESSAGES", "100"))


def _api_key() -> str:
    for name in ("PROXIAPI_KEY", "PROXYAPI_KEY", "OPENAI_API_KEY"):
        value = os.getenv(name, "").strip()
        if value and not value.startswith("your-"):
            return value
    raise RuntimeError(
        "Не найден ключ ProxyAPI. Укажите PROXIAPI_KEY=... в файле agent/.env "
        "(ключ: https://console.proxyapi.ru/keys)"
    )


def _model_name(base_url: str) -> str:
    model = os.getenv("AGENT_MODEL", DEFAULT_MODEL).strip()
    # Эндпоинт /openai/v1 ProxyAPI ждёт имя модели без префикса провайдера: gpt-5-mini
    if "/openai/" in base_url and model.startswith("openai/"):
        model = model.split("/", 1)[1]
    return model


def build_llm() -> ChatOpenAI:
    base_url = os.getenv("PROXYAPI_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    model = _model_name(base_url)
    kwargs: dict[str, Any] = {
        "model": model,
        "api_key": _api_key(),
        "base_url": base_url,
        "timeout": float(os.getenv("AGENT_LLM_TIMEOUT", "120")),
        "max_retries": 2,
    }
    effort = os.getenv("AGENT_REASONING_EFFORT", "low").strip()
    if effort and ("gpt-5" in model or model.split("/")[-1].startswith("o")):
        kwargs["reasoning_effort"] = effort
    log.info("LLM: model=%s base_url=%s reasoning_effort=%s", model, base_url, kwargs.get("reasoning_effort"))
    return ChatOpenAI(**kwargs)


SYSTEM_PROMPT = """Ты — локальный AI-агент, работающий в терминале пользователя. Отвечай на языке пользователя (по умолчанию — по-русски).

Текущие дата и время: {now}.
Рабочая папка для файлов и команд: {workspace}

## Как работать
1. Проанализируй задачу. Сложную задачу разбей на шаги и выполняй их по очереди, вызывая инструменты столько раз, сколько нужно.
2. Выбери инструмент:
   - погода → get_weather
   - цена криптовалюты (bitcoin, ethereum, ...) → get_crypto_price
   - официальный курс валюты ЦБ РФ (доллар, евро, юань...) на дату → get_currency_cb_rate
   - цена акции на Московской бирже → get_share_price
   - нужна информация/факты/новости → web_search (короткий запрос 3-7 слов), затем при необходимости fetch_webpage для чтения найденной страницы
   - обращение к произвольному публичному API → http_request
   - работа с данными в файлах → read_file / write_file / list_directory
   - действие в системе → run_terminal_command (только команды из белого списка)
3. Не выдумывай цены, курсы, погоду и факты — получай их инструментами. Если инструмент вернул ошибку, попробуй исправить параметры или другой инструмент, а если не вышло — честно сообщи об ошибке.
4. Если для выполнения не хватает КРИТИЧНЫХ данных (например, «Какая погода?» без города, и город не известен из памяти) — НЕ вызывай инструменты, а задай короткий уточняющий вопрос.
   - Никогда не спрашивай разрешения воспользоваться интернетом или инструментами — пользователь уже разрешил, просто делай.
   - Не уточняй второстепенные детали (бренд, упаковку, валюту по умолчанию): выбери разумный вариант, выполни задачу и укажи в ответе, какое допущение сделал.
   - Не задавай больше одного уточняющего вопроса подряд.
5. Если нужные данные получить не удалось (сайт не отдаёт цены, поиск пуст после 2-3 переформулировок) — не зацикливайся: сообщи, что удалось найти, что нет и почему, и предложи альтернативный источник.
6. Учитывай память о прошлых разговорах: если пользователь ранее называл свой город, валюту и т.п. — используй это.
7. Относительные даты («вчера», «на прошлой неделе») переводи в конкретные даты по текущей дате.

## Формат ответа
- Коротко и структурировано: главный результат в первой строке, детали — списком.
- Указывай источник данных и момент времени, к которому они относятся.
- Не используй таблицы шире 4 колонок: ответ читается в терминале.

## Память о прошлых разговорах
{summary}
"""

SUMMARY_PROMPT = """Ты ведёшь краткую долговременную память ассистента.
Обнови резюме диалога с учётом нового обмена репликами. Сохраняй: факты о пользователе
(имя, город, предпочтения), важные результаты (курсы, цены, созданные файлы) с датами,
незавершённые задачи. Удаляй устаревшее и несущественное. Не более 15 пунктов, списком, по-русски.

Текущее резюме:
{summary}

Новый обмен:
Пользователь: {user}
Ассистент: {assistant}
Использованные инструменты: {tools}

Верни только обновлённое резюме."""


class Memory:
    """Долговременная память: последние сообщения + накопительное резюме, хранится в JSON."""

    def __init__(self, path: Path = MEMORY_FILE):
        self.path = path
        self.data: dict[str, Any] = {"summary": "", "messages": [], "updated_at": None}
        self.load()

    def load(self) -> None:
        if not self.path.exists() or not self.path.read_text(encoding="utf-8").strip():
            self.save()
            return
        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
            self.data.update({k: loaded.get(k, v) for k, v in self.data.items()})
            log.info("Память загружена: %d сообщений", len(self.data["messages"]))
        except json.JSONDecodeError:
            backup = self.path.with_suffix(f".broken-{int(time.time())}.json")
            self.path.replace(backup)
            log.error("memory.json повреждён, сохранён как %s, начинаю с чистой памяти", backup.name)
            self.save()

    def save(self) -> None:
        self.data["updated_at"] = datetime.now().isoformat(timespec="seconds")
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    @property
    def summary(self) -> str:
        return self.data["summary"]

    def recent_messages(self, n: int = CONTEXT_MESSAGES) -> list[dict[str, str]]:
        return [{"role": m["role"], "content": m["content"]} for m in self.data["messages"][-n:]]

    def add_turn(self, user: str, assistant: str, tools_used: list[str]) -> None:
        ts = datetime.now().isoformat(timespec="seconds")
        self.data["messages"].append({"role": "user", "content": user, "ts": ts})
        self.data["messages"].append(
            {"role": "assistant", "content": assistant, "ts": ts, "tools": tools_used}
        )
        self.data["messages"] = self.data["messages"][-MAX_STORED_MESSAGES:]
        self.save()

    def set_summary(self, summary: str) -> None:
        self.data["summary"] = summary.strip()
        self.save()

    def clear(self) -> None:
        self.data = {"summary": "", "messages": [], "updated_at": None}
        self.save()


def _message_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


class Agent:
    def __init__(self, memory: Memory | None = None):
        self.memory = memory or Memory()
        self.llm = build_llm()
        self.tools = ALL_TOOLS
        log.info("Инструменты: %s", ", ".join(t.name for t in self.tools))

    def _system_prompt(self) -> str:
        return SYSTEM_PROMPT.format(
            now=datetime.now().strftime("%d.%m.%Y %H:%M (%A)"),
            workspace=WORKSPACE_DIR,
            summary=self.memory.summary or "(пока пусто)",
        )

    def ask(self, user_input: str) -> dict[str, Any]:
        """Обработать запрос пользователя. Возвращает {'answer', 'tools', 'elapsed'}."""
        started = time.perf_counter()
        log.info("USER: %s", user_input)

        graph = create_agent(model=self.llm, tools=self.tools, system_prompt=self._system_prompt())
        history = self.memory.recent_messages()
        inputs = {"messages": [*history, {"role": "user", "content": user_input}]}
        result = graph.invoke(inputs, config={"recursion_limit": 30})

        new_messages = result["messages"][len(history) + 1 :]
        tools_used: list[str] = []
        for msg in new_messages:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for call in msg.tool_calls:
                    tools_used.append(call["name"])
                    log.info("AGENT -> %s(%s)", call["name"], json.dumps(call["args"], ensure_ascii=False))
            elif isinstance(msg, ToolMessage):
                log.debug("TOOL RESULT [%s]: %s", msg.name, _message_text(msg)[:500])

        answer = _message_text(result["messages"][-1]).strip() or "(пустой ответ модели)"
        elapsed = time.perf_counter() - started
        log.info("ANSWER (%.1fs, tools=%s): %s", elapsed, tools_used, answer[:500].replace("\n", " "))

        self.memory.add_turn(user_input, answer, tools_used)
        self._update_summary(user_input, answer, tools_used)
        return {"answer": answer, "tools": tools_used, "elapsed": elapsed}

    def _update_summary(self, user: str, answer: str, tools_used: list[str]) -> None:
        try:
            prompt = SUMMARY_PROMPT.format(
                summary=self.memory.summary or "(пусто)",
                user=user,
                assistant=answer[:3000],
                tools=", ".join(tools_used) or "нет",
            )
            summary = _message_text(self.llm.invoke(prompt))
            if summary.strip():
                self.memory.set_summary(summary)
                log.debug("Резюме обновлено: %s", summary.replace("\n", " ")[:500])
        except Exception:
            log.exception("Не удалось обновить резюме диалога (ответ пользователю не затронут)")
