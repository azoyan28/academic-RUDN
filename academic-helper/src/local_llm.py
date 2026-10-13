"""Локальная генерация Qwen3 на Apple Silicon, без внешнего API."""
import os
import re
import threading
from functools import lru_cache
from pathlib import Path

from src.config import DATA

MODEL_ID = "mlx-community/Qwen3-1.7B-4bit"
MODEL_PATH = Path(os.getenv("LOCAL_MODEL_PATH", str(DATA / "models" / "qwen3")))
LOCK = threading.Lock()


def numbers_grounded(generated, evidence):
    """Проверяет буквальное присутствие чисел с сохранением признака процента.

    Это не проверка смысла или причинности. Числа внутри идентификаторов признаков пропускаются.
    """
    def values(text):
        tokens = re.findall(r"(?<![\w-])\d+(?:[.,]\d+)?\s*%?", text)
        return {(float(t.strip().rstrip('%').replace(',', '.').strip()), '%' in t) for t in tokens}
    return values(generated).issubset(values(evidence))


@lru_cache(maxsize=1)
def load_local_model():
    if not (MODEL_PATH / "model.safetensors").exists():
        raise ValueError("Локальная модель не скачана. Запустите python -m src.download_model")
    from mlx_lm import load
    return load(str(MODEL_PATH))


def explain(question, grounded_answer):
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler
    import mlx.core as mx

    model, tokenizer = load_local_model()
    messages = [
        {"role": "system", "content": "Ты русскоязычный учебный помощник. Отвечай кратко, 2–4 предложения. "
         "Отвечай прямо на вопрос. Если запрошен список записей, перечисли номера из справки. "
         "Контекст: все записи студентов созданы генератором для вымышленного российского вуза. "
         "Официальный справочник РУДН реален, но оценки и студенты не относятся к РУДН. "
         "Цель прогноза — итоговая 2 по шкале 2–5. По отчислению ответ всегда отрицательный. "
         "Все дисциплины одного студента находятся в одной выборке. "
         "Опирайся только на справку. Не придумывай факты, значения или причины. Не повторяй ссылки. "
         "Не прогнозируй отчисление. Если в справке нет ответа, скажи, что данных недостаточно."},
        {"role": "user", "content": f"Вопрос: {question}\n\nПроверенная справка:\n{grounded_answer}\n\nОтвет на русском языке:"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    with LOCK:
        mx.random.seed(42)
        return generate(model, tokenizer, prompt=prompt, max_tokens=240,
                        sampler=make_sampler(temp=.7, top_p=.8, top_k=20), verbose=False).strip()
