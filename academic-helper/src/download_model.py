"""Скачивание открытой локальной LLM без аккаунта Hugging Face."""
from src.local_llm import MODEL_ID, MODEL_PATH


def main():
    from huggingface_hub import snapshot_download
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    snapshot_download(MODEL_ID, local_dir=str(MODEL_PATH))
    print(f"Модель сохранена в {MODEL_PATH}. Перезапустите Streamlit.")


if __name__ == "__main__":
    main()
