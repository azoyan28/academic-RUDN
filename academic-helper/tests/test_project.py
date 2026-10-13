import json
import shutil
import sqlite3
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from src.agent import Agent, Tools
from src.config import ARTIFACTS, DATA, DB_PATH, FEATURES
from src.etl import clean_grades, clean_activity, integrate
from src.local_llm import numbers_grounded


@pytest.fixture
def tools(tmp_path):
    db=tmp_path/'test.sqlite';shutil.copy2(DB_PATH,db)
    return Tools(db_path=db)


def test_student_never_crosses_splits():
    splits=pd.read_csv(ARTIFACTS/'splits.csv')
    assert splits.groupby('entity_key').split.nunique().max()==1
    assert splits.record_id.nunique()==1080
    assert 'final_grade' not in FEATURES and 'risk_target' not in FEATURES


def test_all_grades_follow_scale():
    frame=pd.read_csv(DATA/'clean.csv')
    for name in ['control_1','control_2','final_grade']:
        assert set(frame[name].dropna().unique()) <= {2,3,4,5}
    assert frame.entity_key.nunique()==360
    assert frame.groupby('entity_key').course.nunique().eq(3).all()


def test_cleaning_and_origin():
    cleaned,audit=clean_grades(pd.read_csv(DATA/'raw/gradebook.csv'))
    assert audit['duplicate_rows']==5 and len(cleaned)==1080 and audit['missing_cells']==3
    assert cleaned.iloc[5].course=='pr' and cleaned.iloc[5].source_row==7


def test_invalid_grade_rejected():
    raw=pd.read_csv(DATA/'raw/gradebook.csv').head(3).copy()
    raw.loc[0,'final_grade']=6;raw.loc[1,'final_grade']=np.nan
    clean,audit=clean_grades(raw)
    assert len(clean)==1 and audit['rejected_rows']==2


def test_conflicting_grades_rejected():
    raw=pd.read_csv(DATA/'raw/gradebook.csv').head(3).copy()
    extra=raw.iloc[[0]].copy();extra['final_grade']=5 if raw.iloc[0].final_grade!=5 else 2
    with pytest.raises(ValueError):clean_grades(pd.concat([raw,extra],ignore_index=True))


def test_future_events_cannot_change_features():
    grades,_=clean_grades(pd.read_csv(DATA/'raw/gradebook.csv'))
    raw=pd.read_csv(DATA/'raw/activity_log.csv')
    activity,audit=clean_activity(raw)
    assert audit['duplicate_rows']==12 and audit['future_events_excluded']==21600
    first=integrate(grades,activity)
    raw.loc[raw.event_date.gt('2026-03-29') & raw.event_type.eq('attendance'),'attended']=0
    changed,_=clean_activity(raw)
    pd.testing.assert_frame_equal(first,integrate(grades,changed))


def test_activity_status_consistency():
    raw=pd.read_csv(DATA/'raw/activity_log.csv')
    row=raw.loc[raw.event_type.eq('lab')].iloc[[0]].copy()
    row['submitted']=0;row['late']=1
    clean,audit=clean_activity(row)
    assert len(clean)==0 and audit['rejected_rows']==1


def test_aggregation_counts():
    frame=pd.read_csv(DATA/'clean.csv')
    assert frame.classes_total.eq(16).all() and frame.labs_assigned.eq(4).all()
    assert frame.classes_attended.le(frame.classes_total).all()
    assert frame.late_labs.le(frame.labs_submitted).all()
    assert frame.attendance_rate.between(0,1).all()


def test_prediction_ignores_final_grade(tools):
    frame=tools.records('ud').head(3);before=tools.predict(frame)
    frame['final_grade']=5;frame['risk_target']=0
    assert np.array_equal(before,tools.predict(frame))


def test_missing_control_uses_imputer(tools):
    frame=tools.records('bd').head(2).copy();frame['control_2']=np.nan
    assert np.isfinite(tools.predict(frame)).all()


def test_sql_and_allowlist(tools):
    with pytest.raises(ValueError):tools.course_stats("ud';DROP TABLE students;--")
    with pytest.raises(ValueError):tools.predict_record("ud-0001' OR 1=1")
    with pytest.raises(ValueError):tools.call('execute_sql',{'query':'DELETE FROM students'})
    assert sum(r['records'] for r in tools.course_stats())==1080


def test_retrieval_and_abstention(tools):
    docs=tools.search_knowledge('Какая шкала оценок используется?')
    assert 'project-scale' in [r['document_id'] for r in docs]
    assert tools.search_knowledge('zzzxqv')==[]


def test_answer_and_logging(tools):
    r=Agent(tools=tools).ask('Какова средняя оценка по управлению данными?')
    assert r['status']=='ok' and r['tools']==['course_stats'] and '360' in r['answer']
    with sqlite3.connect(tools.db_path) as con:
        row=con.execute('SELECT mode,status FROM requests ORDER BY request_id DESC LIMIT 1').fetchone()
    assert row==('offline','ok')


def test_llm_exchange_without_real_api(tools):
    class FakeResponses:
        def __init__(self):self.calls=[]
        def create(self,**kwargs):
            self.calls.append(kwargs)
            if len(self.calls)==1:
                return SimpleNamespace(output=[SimpleNamespace(type='function_call',name='course_stats',
                    call_id='call1',arguments=json.dumps({'course':'ud'}))],output_text='')
            output=[i for i in kwargs['input'] if isinstance(i,dict) and i.get('type')=='function_call_output']
            assert json.loads(output[0]['output'])[0]['records']==360
            return SimpleNamespace(output=[],output_text='360 записей, данные синтетические.')
    client=SimpleNamespace(responses=FakeResponses())
    assert Agent(mode='llm',tools=tools,client=client,model='mock').ask('Статистика')['status']=='ok'


def test_external_api_failure_is_explicit(tools,monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY',raising=False)
    r=Agent(mode='llm',tools=tools,model='mock').ask('Статистика')
    assert r['status']=='error' and 'OPENAI_API_KEY' in r['answer']


def test_number_guard_and_fallback(tools,monkeypatch):
    assert numbers_grounded('Средняя оценка 3,28','Средняя оценка 3.28')
    assert not numbers_grounded('130% случаев','130 записей, доля 36.1%')
    monkeypatch.setattr('src.local_llm.explain',lambda *_:'Доля 999%')
    r=Agent(mode='local',tools=tools).ask('Сколько записей по базам данных?')
    assert r['generation_accepted'] is False and '999%' not in r['answer']


@pytest.mark.parametrize('page',['Обзор данных','Прогноз','Помощник','Качество модели','Журнал запросов'])
def test_streamlit_pages(page):
    from streamlit.testing.v1 import AppTest
    at=AppTest.from_file(str(DATA.parent/'app.py'),default_timeout=30).run()
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception


def test_streamlit_actions_and_grade_choices():
    from streamlit.testing.v1 import AppTest
    at=AppTest.from_file(str(DATA.parent/'app.py'),default_timeout=30).run()
    at.sidebar.radio[0].set_value('Прогноз').run()
    assert at.selectbox[1].options==['2','3','4','5']
    at.button[0].click().run()
    assert not at.exception and at.metric[0].label=='Скор риска'
    at.sidebar.radio[0].set_value('Помощник').run()
    at.button[0].click().run()
    assert not at.exception and any('360' in p.value for p in at.markdown)


def test_database_relations():
    with sqlite3.connect(DB_PATH) as con:
        assert con.execute('PRAGMA foreign_key_check').fetchall()==[]
        assert con.execute('SELECT COUNT(*) FROM activities').fetchone()[0]==21600
