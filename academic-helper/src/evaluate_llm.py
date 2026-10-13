"""Явно запускаемая пользователем проверка внешней LLM, без подмены режимов."""
import argparse
import json
import os

from src.agent import Agent
from src.config import ARTIFACTS
from src.evaluate_agent import CASES


def main():
    parser = argparse.ArgumentParser(description="Отправляет 16 вопросов и данные функций во внешний API")
    parser.add_argument("--confirm-external", action="store_true", help="Подтверждает запуск внешнего API")
    args = parser.parse_args()
    if not args.confirm_external:
        parser.error("Для внешнего API явно укажите --confirm-external")
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("OPENAI_MODEL"):
        parser.error("Сначала настройте OPENAI_API_KEY и OPENAI_MODEL локально")
    agent, results = Agent(mode="llm"), []
    for question, expected_tool, expected_document in CASES:
        response = agent.ask(question)
        results.append({"question": question, "expected_tool": expected_tool,
                        "expected_document": expected_document, **response})
    report = {"model": os.getenv("OPENAI_MODEL"), "mode": "llm", "real_llm_evaluated": True,
              "successful_requests": sum(r["status"] == "ok" for r in results), "cases": len(results),
              "results": results, "note": "Содержательную правильность ответов проверьте вручную по evidence."}
    (ARTIFACTS / "llm_evaluation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Успешных ответов: {report['successful_requests']} из {len(results)}. Результаты сохранены.")


if __name__ == "__main__":
    main()
