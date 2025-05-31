import os
import re
import json
import asyncio
import time
import numpy as np

from pathlib import Path
from functools import lru_cache
from sentence_transformers import SentenceTransformer, util

from aiogram import Bot, Dispatcher, types
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
        """Загрузка FAQ из JSON файлов в директории"""
        faq_path = Path(self.faq_dir)
        if not faq_path.exists() or not faq_path.is_dir():
            raise SystemExit(f"Директория {self.faq_dir} не найдена!")
        
        print(f"Загрузка FAQ из {self.faq_dir}...")
        for file_path in faq_path.glob("*.json"):
            category = file_path.stem
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    self.faq[category] = json.load(f)
                print(f"Загружено: {category} ( {len(self.faq[category]['questions'])} вопросов)")
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Ошибка в файле {file_path}: {str(e)}")
        
        if not self.faq:
            raise SystemExit("Нет FAQ файлов в директории!")
        
        print(f"Всего загружено {len(self.faq)} категорий FAQ")
    
    def load_model(self):
        """Загрузка модели при первом вызове"""
        if self.model is None:
            self.model = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L12-v2", 
                device="cpu"
            )
    
    def prepare_embeddings(self):
        """Предварительный расчет эмбеддингов для FAQ"""
        for category, data in self.faq.items():
            questions = data["questions"]
            if questions:  # Защита от пустых списков
                self.embeddings_cache[category] = self.model.encode(
                    questions, 
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
    
    @lru_cache(maxsize=200)
    def get_query_embedding(self, text: str) -> np.ndarray:
        """Кэшируем эмбеддинги запросов"""
        return self.model.encode([text])[0]
    
    def is_triggered(self, question: str) -> bool:
        """Проверка триггерных слов"""
        question_lower = question.lower()
        return any(
            trigger in question_lower
            for data in self.faq.values()
            for trigger in data["triggers"]
        )
    
    def find_best_match(self, question: str, threshold: float = 0.65) -> str:
        """Поиск лучшего совпадения"""
        if not self.is_triggered(question):
            return None
            
        start_time = time.time()
        query_embed = self.get_query_embedding(question)
        best_match = None
        best_score = threshold
        
        for category, emb in self.embeddings_cache.items():
            scores = util.cos_sim(query_embed, emb)[0]
            max_score = max(scores).item()
            if max_score > best_score:
                best_score = max_score
                best_match = self.faq[category]["answer"]
        
        print(f"Поиск занял: {time.time()-start_time:.3f} секунд | Сходство: {best_score:.2f}")
        return best_match
    
    def clean_question(self, text: str) -> str:
        """Очистка вопроса"""
        text = re.sub(r"[^\w\s]", "", text)  # Удаляем спецсимволы
        text = re.sub(r"\s+", " ", text).strip().lower()  # Нормализация пробелов
        return text

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