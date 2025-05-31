import os
import re
import json
import asyncio
import time
import numpy as np

from pathlib import Path
from functools import lru_cache
from sentence_transformers import SentenceTransformer, util

from aiogram import Bot, Dispatcher
from aiogram.types import (
    InlineQuery, InlineQueryResultArticle,
    InputTextMessageContent, InlineKeyboardButton,
    InlineKeyboardMarkup, Message )
from aiogram.filters import CommandStart, Command
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode

from config import TOKEN

# Конфигурация
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Отключаем oneDNN оптимизации
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # Снижаем уровень логов TensorFlow

class FAQ_flipper_Bot:
    def __init__(self, faq_dir="faq"):
        self.model = None
        self.faq = {}
        self.embeddings_cache = {}
        self.faq_dir = faq_dir
        
        self.load_faq()
        self.load_model()
        self.prepare_embeddings()
    
    def load_faq(self):
        """Загрузка FAQ без обработки триггеров"""
        faq_path = Path(self.faq_dir)
        print(f"Загрузка FAQ из {self.faq_dir}...")
        for file_path in faq_path.glob("*.json"):
            category = file_path.stem
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.faq[category] = {
                    "questions": data["questions"],
                    "answer": data["answer"]
                }
            print(f"Загружено: {category} ({len(data['questions'])} вопросов)")
        
        print(f"Всего загружено {len(self.faq)} категорий FAQ")
    
    def load_model(self):
        if self.model is None:
            self.model = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L12-v2", 
                device="cpu"
            )
    
    def prepare_embeddings(self):
        """Кэшируем эмбеддинги для всех вопросов"""
        all_questions = []
        self.question_map = {}  # Для связи вопроса с ответом
        
        for category, data in self.faq.items():
            for q in data["questions"]:
                clean_q = self.clean_question(q)
                all_questions.append(clean_q)
                # Сохраняем связь: вопрос -> ответ
                self.question_map[clean_q] = data["answer"]
        
        # Массив эмбеддингов для всех вопросов
        self.all_embeddings = self.model.encode(
            all_questions, 
            convert_to_numpy=True,
            show_progress_bar=False
        )
        print(f"Рассчитано эмбеддингов: {len(self.all_embeddings)}")
    
    @lru_cache(maxsize=500)
    def get_query_embedding(self, text: str) -> np.ndarray:
        return self.model.encode([text], show_progress_bar=False)[0]
    
    def find_best_match(self, question: str, threshold: float = 0.6) -> str:
        """Прямой поиск по семантическому сходству"""
        start_time = time.time()
        clean_question = self.clean_question(question)
        query_embed = self.get_query_embedding(clean_question)
        
        # Сравнение со всеми вопросами
        scores = util.cos_sim(query_embed, self.all_embeddings)[0]
        max_idx = np.argmax(scores).item()
        max_score = scores[max_idx].item()
        
        print(f"Поиск занял: {time.time()-start_time:.3f}с | Сходство: {max_score:.2f}")
        return self.question_map[
            list(self.question_map.keys())[max_idx]
        ] if max_score > threshold else None
    
    def clean_question(self, text: str) -> str:
        text = re.sub(r"[^\w\s]", "", text.lower())
        return re.sub(r"\s+", " ", text).strip()

faq_bot = FAQ_flipper_Bot(faq_dir="faq")
bot = Bot(
        token=TOKEN,
        default=DefaultBotProperties(
            link_preview_is_disabled=True,
            parse_mode=ParseMode.HTML
            )
        )
dp = Dispatcher()

@dp.message(CommandStart())
async def start_command(message: Message):
    msg = """
Приветствую, я бот–нейросеть для твоих вопросов по поводу <b>Flipper Zero</b>.
<i>Нажми кнопку ниже и задай интересующий тебя вопрос</i>.
    """
    await message.answer(
        text=msg,
        reply_markup=
            InlineKeyboardMarkup(
                inline_keyboard=[[
                    InlineKeyboardButton(
                        text="🔍 Задать вопрос",
                        switch_inline_query_current_chat=''
                    )
                ]]
            )
        )

@dp.message(Command("info"))
async def info_command(message: Message):
    await message.answer(text="инфо")

@dp.inline_query()
async def handle_inline_query(inline_query: InlineQuery):
    query = inline_query.query.strip()
    if not query:
        return
    
    # Очищаем и обрабатываем вопрос
    cleaned_query = faq_bot.clean_question(query)
    answer = faq_bot.find_best_match(cleaned_query)
    
    results = []
    if answer:
        # Формируем успешный результат
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
        # Результат когда ответ не найден
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

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())