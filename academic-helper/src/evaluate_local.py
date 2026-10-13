"""Реальная проверка локальной языковой модели с сохранением ответов."""
import json
import time

import numpy as np

from src.agent import Agent
from src.config import ARTIFACTS
from src.evaluate_agent import CASES
from src.local_llm import MODEL_ID, load_local_model


def main():
    started = time.perf_counter()
    load_local_model()
    loading_ms = (time.perf_counter() - started) * 1000
    agent, results = Agent(mode="local"), []
    for question, expected_tool, expected_document in CASES:
        result = agent.ask(question)
        generated = result["raw_generated"]
        results.append({"question": question, "expected_tool": expected_tool,
                        "expected_document": expected_document, "generated_answer": generated, **result})
        print(question, result["status"], f"{result['latency_ms']:.0f} ms", flush=True)
    times = [r["latency_ms"] for r in results]
    report = {"mode": "local", "model": MODEL_ID, "quantization_bits": 4,
              "real_llm_evaluated": True, "cases": len(results), "load_time_ms": loading_ms,
              "successful_requests": sum(r["status"] == "ok" for r in results),
              "mean_latency_ms": float(np.mean(times)), "p95_latency_ms": float(np.quantile(times, .95)),
              "tool_selection": "deterministic_router", "results": results,
              "limitation": "Факт успешной генерации не означает содержательной правильности ответа. Проверьте generated_answer по evidence."}
    (ARTIFACTS / "local_llm_evaluation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k != "results"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
