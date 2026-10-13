"""Интеграция ведомости и журнала на фиксированный момент прогноза."""
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
import pandas as pd
from src.config import AS_OF, COURSES, DATA, DB_PATH, add_features


def clean_grades(frame):
    frame = frame.copy()
    required = ['student_id','group_code','course','semester','control_1','control_2','final_grade']
    if set(required)-set(frame):
        raise ValueError('Ведомость не содержит обязательные поля')
    initial, missing = len(frame), int(frame.isna().sum().sum())
    frame['source_row'] = frame.index + 2
    for c in ['student_id','group_code','course']:
        frame[c]=frame[c].astype('string').str.strip()
    for c in ['semester','control_1','control_2','final_grade']:
        frame[c]=pd.to_numeric(frame[c],errors='coerce')
    valid = frame.student_id.str.fullmatch(r'S\d{4}').fillna(False) & frame.course.isin(COURSES)
    valid &= frame.group_code.notna() & frame.group_code.ne('') & frame.semester.eq(8)
    valid &= frame.final_grade.isin([2,3,4,5])
    for c in ['control_1','control_2']:
        valid &= frame[c].isna() | frame[c].isin([2,3,4,5])
    rejected = int((~valid).sum())
    frame = frame.loc[valid].copy()
    duplicate = frame.duplicated(required)
    duplicates=int(duplicate.sum());frame=frame.loc[~duplicate].copy()
    if frame.duplicated(['student_id','course','semester']).any():
        raise ValueError('Конфликтующие строки для одного студента и дисциплины')
    if frame.groupby('student_id').group_code.nunique().max()>1:
        raise ValueError('Конфликт группы студента')
    return frame,dict(input_rows=initial,missing_cells=missing,rejected_rows=rejected,
                      duplicate_rows=duplicates,output_rows=len(frame))


def clean_activity(frame, as_of=AS_OF):
    frame=frame.copy()
    required=['event_id','student_id','course','event_date','event_type','attended','submitted','late']
    if set(required)-set(frame):raise ValueError('Журнал не содержит обязательные поля')
    initial=len(frame)
    for c in ['event_id','student_id','course','event_type']:
        frame[c]=frame[c].astype('string').str.strip()
    frame['event_date']=pd.to_datetime(frame.event_date,errors='coerce')
    for c in ['attended','submitted','late']:frame[c]=pd.to_numeric(frame[c],errors='coerce')
    valid=frame.student_id.str.fullmatch(r'S\d{4}').fillna(False) & frame.course.isin(COURSES)
    valid &= frame.event_id.notna() & frame.event_date.notna() & frame.event_type.isin(['attendance','lab'])
    a=frame.event_type.eq('attendance');l=frame.event_type.eq('lab')
    valid &= (~a) | frame.attended.isin([0,1])
    valid &= (~l) | (frame.submitted.isin([0,1]) & frame.late.isin([0,1]) & frame.late.le(frame.submitted))
    rejected=int((~valid).sum());frame=frame.loc[valid].copy()
    duplicates=int(frame.duplicated(required).sum());frame=frame.drop_duplicates(required)
    if frame.event_id.duplicated().any():raise ValueError('Конфликт event_id')
    future=int(frame.event_date.gt(pd.Timestamp(as_of)).sum())
    frame=frame.loc[frame.event_date.le(pd.Timestamp(as_of))].copy()
    return frame,dict(input_rows=initial,rejected_rows=rejected,duplicate_rows=duplicates,
                      future_events_excluded=future,output_rows=len(frame),as_of=as_of)


def integrate(grades, activity):
    keys=['student_id','course']
    known=set(map(tuple,grades[keys].to_numpy()))
    if any(tuple(row) not in known for row in activity[keys].to_numpy()):
        raise ValueError('Журнал содержит событие без строки ведомости')
    a=activity[activity.event_type.eq('attendance')].groupby(keys).agg(
        classes_total=('event_id','size'),classes_attended=('attended','sum'))
    l=activity[activity.event_type.eq('lab')].groupby(keys).agg(
        labs_assigned=('event_id','size'),labs_submitted=('submitted','sum'),late_labs=('late','sum'))
    joined=grades.merge(a.reset_index(),on=keys,how='left',validate='one_to_one').merge(
        l.reset_index(),on=keys,how='left',validate='one_to_one')
    if joined[['classes_total','labs_assigned']].isna().any().any():
        raise ValueError('Нет наблюдений посещаемости или работ по дисциплине')
    joined['entity_key']=joined.student_id
    joined['record_id']=joined.course+'-'+joined.student_id.str[1:]
    joined['risk_target']=joined.final_grade.eq(2).astype(int)
    joined['as_of']=AS_OF
    return add_features(joined)


SCHEMA='''
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS sources(source_id TEXT PRIMARY KEY,title TEXT,url TEXT,sha256 TEXT,registered_at TEXT);
CREATE TABLE IF NOT EXISTS students(entity_key TEXT PRIMARY KEY,group_code TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS enrollments(
 record_id TEXT PRIMARY KEY,entity_key TEXT REFERENCES students(entity_key),course TEXT,
 source_id TEXT REFERENCES sources(source_id),source_row INTEGER,
 control_1 REAL,control_2 REAL,final_grade INTEGER CHECK(final_grade BETWEEN 2 AND 5),
 risk_target INTEGER CHECK(risk_target IN (0,1)),as_of TEXT,features_json TEXT);
CREATE INDEX IF NOT EXISTS ix_course ON enrollments(course);
CREATE TABLE IF NOT EXISTS activities(event_id TEXT PRIMARY KEY,record_id TEXT REFERENCES enrollments(record_id),
 source_id TEXT REFERENCES sources(source_id),event_date TEXT,event_type TEXT,attended REAL,submitted REAL,late REAL);
CREATE TABLE IF NOT EXISTS documents(document_id TEXT PRIMARY KEY,source_id TEXT REFERENCES sources(source_id),title TEXT,text TEXT);
CREATE TABLE IF NOT EXISTS experiments(experiment_id INTEGER PRIMARY KEY AUTOINCREMENT,created_at TEXT,
 model_name TEXT,split_name TEXT,metrics_json TEXT);
CREATE TABLE IF NOT EXISTS requests(request_id INTEGER PRIMARY KEY AUTOINCREMENT,created_at TEXT,mode TEXT,
 tool_names TEXT,question TEXT,answer TEXT,latency_ms REAL,status TEXT);
'''


def build_database(frame, activity, docs, sources, db_path=DB_PATH):
    with sqlite3.connect(db_path) as con:
        con.executescript(SCHEMA)
        for table in ['activities','documents','enrollments','students','sources']:
            con.execute(f'DELETE FROM {table}')
        for s in sources:
            con.execute('INSERT INTO sources VALUES (?,?,?,?,?)',(s['id'],s['title'],s['url'],s['sha256'],s['registered_at']))
        for _,r in frame.drop_duplicates('entity_key').iterrows():
            con.execute('INSERT INTO students VALUES (?,?)',(r.entity_key,r.group_code))
        for _,r in frame.iterrows():
            d={k:None if pd.isna(v) else v for k,v in r.to_dict().items()}
            con.execute('INSERT INTO enrollments VALUES (?,?,?,?,?,?,?,?,?,?,?)',(
                r.record_id,r.entity_key,r.course,'gradebook',int(r.source_row),
                r.control_1,r.control_2,int(r.final_grade),int(r.risk_target),AS_OF,json.dumps(d,ensure_ascii=False)))
        for _,r in activity.iterrows():
            def optional(v):return None if pd.isna(v) else float(v)
            con.execute('INSERT INTO activities VALUES (?,?,?,?,?,?,?,?)',(
                r.event_id,r.course+'-'+r.student_id[1:],'activity',r.event_date.strftime('%Y-%m-%d'),
                r.event_type,optional(r.attended),optional(r.submitted),optional(r.late)))
        for d in docs:
            con.execute('INSERT INTO documents VALUES (?,?,?,?)',(d['id'],d['source_id'],d['title'],d['text']))


def run_etl():
    if not (DATA/'raw/gradebook.csv').exists():
        from src.generate_data import generate
        generate()
    g,ga=clean_grades(pd.read_csv(DATA/'raw/gradebook.csv'))
    a,aa=clean_activity(pd.read_csv(DATA/'raw/activity_log.csv'))
    frame=integrate(g,a)
    docs=json.loads((DATA/'knowledge/notes.json').read_text())
    registered=datetime.now(timezone.utc).isoformat()
    specs=[('gradebook','Синтетическая ведомость оценок','raw/gradebook.csv'),
           ('activity','Синтетический журнал занятий и работ','raw/activity_log.csv'),
           ('project','Учебный регламент вымышленного вуза','knowledge/notes.json'),
           ('rudn','Официальный справочник РУДН о системе оценивания','knowledge/rudn_summary.txt')]
    sources=[dict(id=i,title=t,url='https://handbook.rudn.ru/learn/rating.html' if i=='rudn' else 'local:data/'+p,
             sha256=hashlib.sha256((DATA/p).read_bytes()).hexdigest(),registered_at=registered) for i,t,p in specs]
    build_database(frame,a,docs,sources)
    frame.to_csv(DATA/'clean.csv',index=False)
    report=dict(is_synthetic=True,country_context='Россия',scale=[2,3,4,5],as_of=AS_OF,sources=sources,
        audits={'gradebook':ga,'activity':aa},records=len(frame),profile_groups=int(frame.entity_key.nunique()),
        shared_profile_groups=int(frame.groupby('entity_key').course.nunique().gt(1).sum()),
        risk_count=int(frame.risk_target.sum()),activity_records=len(a))
    (DATA/'quality_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    return frame,report


if __name__=='__main__':
    _,report=run_etl();print(json.dumps(report,ensure_ascii=False,indent=2))
