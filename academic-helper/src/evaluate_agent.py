"""Фиксированные сценарии российской учебной версии."""
import json
import numpy as np
from src.agent import Agent, Tools
from src.config import ARTIFACTS

CASES = [
 ('Какова средняя оценка по управлению данными?','course_stats',None),
 ('Сколько записей по базам данных?','course_stats',None),
 ('Какова доля двоек по программированию?','course_stats',None),
 ('Покажи список записей повышенного риска по управлению данными','list_risk',None),
 ('Покажи список группы риска по базам данных','list_risk',None),
 ('Оцени риск для ud-0001','predict_record',None),
 ('Дай прогноз для pr-0010','predict_record',None),
 ('Какая шкала оценок используется?','search_knowledge','project-scale'),
 ('Что означает итоговая двойка?','search_knowledge','project-scale'),
 ('Как рассчитывается посещаемость?','search_knowledge','project-attendance'),
 ('Как рассчитываются доли сдачи работ и просрочек?','search_knowledge','project-labs'),
 ('Когда строится прогноз и как исключаются будущие данные?','search_knowledge','project-cutoff'),
 ('Откуда взяты данные, являются ли они синтетическими?','search_knowledge','project-synthetic'),
 ('Как разделяются студенты между выборками?','search_knowledge','project-split'),
 ('Что можно сделать для подготовки и сдачи работ?','search_knowledge','project-support'),
 ('Можно ли применять модель для реального отчисления?','search_knowledge','project-limits'),
]


def evaluate():
    tools,results=Tools(),[]
    agent=Agent(tools=tools)
    for question,expected_tool,document_id in CASES:
        result=agent.ask(question)
        ids=[d['document_id'] for d in result['evidence']] if result['tools']==['search_knowledge'] else []
        results.append(dict(question=question,expected_tool=expected_tool,actual_tools=result['tools'],
            tool_correct=result['tools']==[expected_tool],expected_document=document_id,
            retrieved_documents=ids,retrieval_hit_at_3=document_id in ids if document_id else None,
            answer=result['answer'],latency_ms=result['latency_ms'],status=result['status']))
    rag=[r for r in results if r['expected_document']]
    times=[r['latency_ms'] for r in results]
    report=dict(mode='offline',real_llm_evaluated=False,cases=len(results),
        tool_routing_accuracy=float(np.mean([r['tool_correct'] for r in results])),
        retrieval_hit_at_3=float(np.mean([r['retrieval_hit_at_3'] for r in rag])),
        retrieval_cases=len(rag),mean_latency_ms=float(np.mean(times)),
        p95_latency_ms=float(np.quantile(times,.95)),results=results,
        limitation='Малый разработческий набор. Не является независимой оценкой свободного диалога.')
    (ARTIFACTS/'agent_evaluation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='results'},ensure_ascii=False,indent=2))


if __name__=='__main__':evaluate()
