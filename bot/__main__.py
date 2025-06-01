import logging
import asyncio
import json
import os
import re
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer, util

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart, Command
from aiogram.types import (
    InlineQuery, InlineQueryResultArticle,
    InputTextMessageContent, InlineKeyboardButton,
    InlineKeyboardMarkup, Message
)

from bot.config import TOKEN

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

START_TIME = time.time()

logger = logging.getLogger(__name__)

class FaqFlipperBot:

    def __init__(self, faq_dir="faq") -> None:
        self.model = None
        self.faq = {}
        self.embeddings_cache = {}
        self.faq_dir = faq_dir

        self.load_faq()
        self.load_model()
        self.prepare_embeddings()

    def load_faq(self) -> None:
        """Загрузка FAQ из JSON-файлов с обработкой ошибок"""
        faq_path = Path(self.faq_dir)
        print(f"Загрузка FAQ из {self.faq_dir}...")
        error_files = []

        for file_path in faq_path.glob("*.json"):
            category = file_path.stem
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                if "questions" not in data or "answer" not in data:
                    raise ValueError(f"Некорректная структура файла {file_path.name}")
                
                self.faq[category] = {
                    "questions": data["questions"],
                    "answer": data["answer"]
                }
                print(f"Загружено: {category} ({len(data['questions'])} вопросов)")
            
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Ошибка в файле {file_path.name}: {str(e)}")
                error_files.append(file_path.name)
            except Exception as e:
                print(f"Неизвестная ошибка в {file_path.name}: {str(e)}")
                error_files.append(file_path.name)

        print(f"\nВсего загружено: {len(self.faq)} корректных файлов")
        if error_files:
            print(f"Файлы с ошибками ({len(error_files)}):")
            for fname in error_files:
                print(f"  - {fname}")

    def load_model(self):
        """Загружает модель SentenceTransformer"""

        if self.model is None:

            self.model = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L12-v2",
                device="cpu"
            )

    def prepare_embeddings(self):
        """Кэшируем эмбеддинги для всех вопросов"""
        all_questions = []
        self.question_map = {}

        for category, data in self.faq.items():
            for q in data["questions"]:
                clean_q = self.clean_question(q)
                all_questions.append(clean_q)
                self.question_map[clean_q] = data["answer"]

        self.all_embeddings = self.model.encode(
            all_questions,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        print(f"Рассчитано эмбеддингов: {len(self.all_embeddings)}")

    @lru_cache(maxsize=500)
    def get_query_embedding(self, text: str) -> np.ndarray:
        """Получает эмбеддинг текстового запроса"""
        return self.model.encode([text], show_progress_bar=False)[0]

    def find_best_match(self, question: str, threshold: float = 0.6) -> str:
        """Прямой поиск по семантическому сходству"""
        start_time = time.time()
        clean_question = self.clean_question(question)
        query_embed = self.get_query_embedding(clean_question)

        scores = util.cos_sim(query_embed, self.all_embeddings)[0]
        max_idx = np.argmax(scores).item()
        max_score = scores[max_idx].item()

        logger.info(f"Вопрос {question} | Поиск занял: {time.time()-start_time:.3f}с | Сходство: {max_score:.2f}")
        return self.question_map[
            list(self.question_map.keys())[max_idx]
        ] if max_score > threshold else None

    def clean_question(self, text: str) -> str:
        text = re.sub(r"[^\w\s]", "", text.lower())
        return re.sub(r"\s+", " ", text).strip()


faq_bot = FaqFlipperBot(faq_dir="faq")
bot = Bot(token=TOKEN, default=DefaultBotProperties(link_preview_is_disabled=True, parse_mode=ParseMode.HTML))
dp = Dispatcher()


@dp.message(CommandStart())
async def start_command(message: Message) -> None:
    msg = """
Приветствую, я бот–нейросеть для твоих вопросов по поводу <b>Flipper Zero</b>.
<i>Нажми кнопку ниже и задай интересующий тебя вопрос</i>.
    """
    await message.answer(
        text=msg,
        reply_markup=InlineKeyboardMarkup(
                inline_keyboard=[[InlineKeyboardButton(text="🔍 Задать вопрос", switch_inline_query_current_chat="")
                ]]
            )
        )


@dp.message(Command("info"))
async def info_command(message: Message) -> None:

    uptime_seconds = int(time.time() - START_TIME)
    days, remainder = divmod(uptime_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)

    uptime_str = f"<i>{days}</i> Д. <i>{hours}</i> Ч. <i>{minutes}</i> Мин."

    info_text = f"""
🤖 <i>Информация о боте</i>

• <i>Аптайм:</i> {uptime_str}
• <i>Категорий FAQ:</i> <code>{len(faq_bot.faq)}</code>
• <i>Всего вопросов:</i> <code>{len(faq_bot.question_map)}</code>

<i>GitHub:</i> <i>github.com/FlipperAI-Lab/flipper-faq-ai</i>
    """

    await message.answer(text=info_text)


@dp.inline_query()
async def handle_inline_query(inline_query: InlineQuery) -> None:
    query = inline_query.query.strip()

    if not query:

        return

    cleaned_query = faq_bot.clean_question(query)
    answer = faq_bot.find_best_match(cleaned_query)

    results = []

    if answer:

        item = InlineQueryResultArticle(
            id="1",
            title="✅ Ответ найден!",
            input_message_content=InputTextMessageContent(
                message_text=answer
            ),
            description=answer[:100] + "..." if len(answer) > 100 else answer
        )
        results.append(item)

    else:

        item = InlineQueryResultArticle(
            id="1",
            title="❌ Ответ не найден",
            input_message_content=InputTextMessageContent(
                message_text="Извините, я не нашел ответа на ваш вопрос."
            ),
            description="Попробуйте переформулировать запрос"
        )
        results.append(item)

    await inline_query.answer(results, is_personal=True, cache_time=1)


async def main() -> None:
    await dp.start_polling(bot)

if __name__ == "__main__":

    try:

        asyncio.run(main())

    except KeyboardInterrupt:

        logger.warning("Бот остановлен пользователем")
