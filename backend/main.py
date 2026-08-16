"""
Прототип: интеграция ИИ с данными бронирования.

Один эндпоинт POST /api/chat принимает сообщение гостя и текущий статус
паспорта, строит system prompt с бизнес-правилами отеля и отправляет
запрос в Groq API (openai/gpt-oss-20b). Ответ модели возвращается на
фронтенд как есть — никаких заранее записанных фраз в коде нет.
"""

import os
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from groq import Groq
from groq import APIError

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("booking-ai")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

if not GROQ_API_KEY:
    logger.warning(
        "GROQ_API_KEY не задан. Создайте .env на основе .env.example."
    )
    client = None
else:
    client = Groq(api_key=GROQ_API_KEY)

app = FastAPI(title="Booking AI Prototype")

# В проде здесь должен быть конкретный origin фронтенда
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    passport_status: str = Field(..., pattern="^(received|not_received)$")


class ChatResponse(BaseModel):
    reply: str


def build_system_prompt(passport_status: str) -> str:
    """Формирует system prompt с бизнес-правилами отеля.

    Статус паспорта подставляется на бэкенде — фронтенд не может
    подделать инструкцию модели, только сырое значение статуса.
    """
    status_ru = "получен" if passport_status == "received" else "не получен"

    return f"""Ты — виртуальный ассистент ресепшена отеля. Твоя задача — отвечать
гостям на вопросы о заселении, опираясь СТРОГО на текущий статус их
бронирования.

Текущий статус паспорта гостя: {status_ru}

Правила:
1. Если статус "не получен" — объясни гостю, что для заселения сначала
   нужно предоставить паспорт. Дай ссылку: https://example.com/passport.
   Не упоминай оплату залога, пока паспорт не получен.
2. Если статус "получен" — сообщи, что паспорт принят, и следующий шаг —
   оплата залога. Не проси паспорт повторно.
3. Отвечай кратко (2-4 предложения), дружелюбно и по-деловому, на русском языке.
4. Если вопрос гостя не связан с заселением, вежливо верни разговор к теме
   бронирования и статусу паспорта.
5. Не выдумывай информацию о цене, датах или номере, которых нет в этом
   контексте."""


@app.post("/api/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    if not GROQ_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY не настроен на сервере.",
        )

    system_prompt = build_system_prompt(payload.passport_status)

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": payload.message},
            ],
            temperature=1,
            max_completion_tokens=300,
            top_p=1,
            stream=False,
        )
    except APIError as exc:
        logger.exception("Ошибка вызова Groq API")
        raise HTTPException(status_code=502, detail=f"Ошибка LLM API: {exc}")

    reply_text = (response.choices[0].message.content or "").strip()

    if not reply_text:
        raise HTTPException(status_code=502, detail="Пустой ответ от модели.")

    return ChatResponse(reply=reply_text)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "model": GROQ_MODEL}


# Раздаём фронтенд как статику той же командой uvicorn.
frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")