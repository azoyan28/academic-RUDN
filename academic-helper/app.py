"""Российская учебная версия, шкала 2–5. Запуск streamlit run app.py."""
import json
import os
import importlib.util
import platform
import sqlite3
import pandas as pd
import streamlit as st
from src.agent import Agent, Tools
from src.config import ARTIFACTS, DB_PATH, COURSES

st.set_page_config(page_title='Учебный помощник России',page_icon='📚',layout='wide')
st.title('Помощник по успеваемости')
st.caption('Учебный институт «Южный кампус» · Россия · оценки 2–5')
st.info('Все записи созданы генератором. Вуз вымышленный. Это не реальные данные РУДН и не статистика России.')
if not DB_PATH.exists() or not (ARTIFACTS/'model.joblib').exists():
    st.info('Подготовьте проект: python -m src.train');st.stop()


@st.cache_resource
def get_tools():return Tools()


tools=get_tools()
metrics=json.loads((ARTIFACTS/'metrics.json').read_text())
page=st.sidebar.radio('Раздел',['Обзор данных','Прогноз','Помощник','Качество модели','Журнал запросов'])
label=st.sidebar.selectbox('Дисциплина',['Все дисциплины',*COURSES.values()])
course=next((k for k,v in COURSES.items() if v==label),None)
st.sidebar.caption('Момент прогноза: конец восьмой недели. События после 29.03.2026 исключены.')
st.sidebar.caption('Цель: итоговая оценка 2. Один студент может иметь записи по трём дисциплинам.')

if page=='Обзор данных':
    frame=tools.records(course)
    cols=st.columns(4)
    cols[0].metric('Записи по дисциплинам',len(frame))
    cols[1].metric('Студенты',frame.student_id.nunique())
    cols[2].metric('Средняя итоговая оценка',f'{frame.final_grade.mean():.2f}')
    cols[3].metric('Доля итоговых двоек',f'{frame.risk_target.mean():.1%}')
    st.subheader('Итоговые оценки от 2 до 5')
    counts=frame.groupby(['final_grade','course']).size().unstack(fill_value=0).reindex([2,3,4,5],fill_value=0)
    counts=counts.rename(columns=COURSES)
    st.bar_chart(counts,x_label='Итоговая оценка',y_label='Записи')
    st.subheader('Статистика по дисциплинам')
    stats=pd.DataFrame(tools.course_stats(course));stats['course']=stats.course.map(COURSES)
    st.dataframe(stats,hide_index=True,width='stretch')
    st.subheader('Данные к середине семестра')
    view=frame[['record_id','student_id','group_code','course','control_1','control_2',
        'attendance_rate','lab_completion','late_share','final_grade']]
    st.dataframe(view,hide_index=True,width='stretch')
    st.download_button('Скачать выбранные записи',view.to_csv(index=False).encode('utf-8-sig'),'records.csv','text/csv')
elif page=='Прогноз':
    st.subheader('Риск итоговой двойки')
    st.write('Введите сведения первой половины семестра. Оценки принимают только значения 2, 3, 4, 5.')
    with st.form('prediction'):
        left,right=st.columns(2)
        selected=left.selectbox('Дисциплина для прогноза',list(COURSES),format_func=COURSES.get)
        c1=left.selectbox('Контрольная 1',[2,3,4,5],index=1)
        c2=left.selectbox('Контрольная 2',[2,3,4,5],index=1)
        total=right.number_input('Проведено занятий',min_value=1,max_value=100,value=16)
        attended=right.number_input('Посещено занятий',min_value=0,max_value=100,value=12)
        assigned=left.number_input('Назначено лабораторных работ',min_value=1,max_value=30,value=4)
        submitted=right.number_input('Сдано лабораторных работ',min_value=0,max_value=30,value=3)
        late=right.number_input('Из них с опозданием',min_value=0,max_value=30,value=1)
        submit=st.form_submit_button('Рассчитать риск')
    if submit:
        if attended>total or submitted>assigned or late>submitted:
            st.error('Число посещённых занятий, сданных работ или просрочек превышает допустимое.')
        else:
            values=dict(course=selected,control_1=c1,control_2=c2,classes_total=total,
                classes_attended=attended,labs_assigned=assigned,labs_submitted=submitted,late_labs=late)
            score=float(tools.predict(pd.DataFrame([values]))[0])
            st.metric('Скор риска',f'{score:.1%}')
            st.write(f"Порог {tools.bundle['threshold']:.1%}. Сигнал повышенного риска: {'да' if score>=tools.bundle['threshold'] else 'нет'}.")
            st.caption('Оценка относится к синтетической модели. Скор требует проверки калибровки на реальных данных.')
elif page=='Помощник':
    st.subheader('Вопросы к данным и справочным материалам')
    choices=['Автономный режим']
    from src.local_llm import MODEL_PATH
    if platform.machine()=='arm64' and importlib.util.find_spec('mlx_lm') and (MODEL_PATH/'model.safetensors').exists():
        choices.append('Локальная LLM')
    if os.getenv('OPENAI_API_KEY') and os.getenv('OPENAI_MODEL'):choices.append('Внешняя LLM')
    label=st.radio('Режим',choices,horizontal=True)
    mode={'Автономный режим':'offline','Локальная LLM':'local','Внешняя LLM':'llm'}[label]
    st.caption('Локальная LLM объясняет результаты функций и поиска. Проверенные данные показаны под ответом.')
    examples=['Какова средняя оценка по управлению данными?','Покажи список группы риска по базам данных',
        'Оцени риск для ud-0001','Какая шкала оценок используется?','Откуда взяты синтетические данные?']
    example=st.selectbox('Пример вопроса',examples)
    question=st.text_input('Ваш вопрос',value=example,max_chars=2000)
    if st.button('Получить ответ',type='primary'):
        with st.spinner('Проверяю данные'):result=Agent(mode=mode,tools=tools).ask(question)
        if result['status']=='ok':st.markdown(result['answer'])
        else:st.error(result['answer'])
        st.caption(f"Время ответа {result['latency_ms']:.0f} мс. Инструменты: {', '.join(result['tools']) or 'нет'}.")
        with st.expander('Данные, использованные для ответа'):st.json(result['evidence'])
elif page=='Качество модели':
    st.subheader('Оценка на отложенных синтетических данных')
    st.write('Одни и те же студенты не пересекаются между выборками. Метрики проверяют зависимости генератора, а не качество на реальных студентах России.')
    result=metrics['test'][metrics['selected_model']]
    cols=st.columns(4)
    for col,name in zip(cols,['f1','precision','recall','roc_auc']):col.metric(name,f'{result[name]:.3f}')
    st.write(f"Тест {result['n']} записей. Модель {metrics['selected_model']}. Порог {metrics['threshold']:.3f}.")
    st.dataframe(pd.DataFrame(metrics['test']).T[['accuracy','precision','recall','f1','roc_auc','average_precision','brier']].round(3),width='stretch')
    st.image(str(ARTIFACTS/'figures/confusion.png'),width=600)
    st.image(str(ARTIFACTS/'figures/importance.png'),width=850)
else:
    st.subheader('Активность помощника')
    with sqlite3.connect(DB_PATH) as con:
        log=pd.read_sql_query('SELECT * FROM requests ORDER BY request_id DESC LIMIT 200',con)
        summary=con.execute("SELECT COUNT(*),AVG(latency_ms),SUM(status='error') FROM requests").fetchone()
    cols=st.columns(3)
    cols[0].metric('Запросы',summary[0]);cols[1].metric('Среднее время',f'{summary[1] or 0:.1f} мс');cols[2].metric('Ошибки',summary[2] or 0)
    if len(log):
        st.bar_chart(log.groupby('mode').size())
        st.dataframe(log[['created_at','mode','question','tool_names','latency_ms','status']],hide_index=True,width='stretch')
    else:st.info('Пока нет обращений. Задайте вопрос помощнику.')
