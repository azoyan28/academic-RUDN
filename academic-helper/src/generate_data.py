"""Синтетический вуз России. Наблюдения не относятся к реальным людям."""
import json
from datetime import date, timedelta
import numpy as np
import pandas as pd
from src.config import AS_OF, COURSES, DATA


def grade(value):
    # Пороги генератора не являются правилом реального вуза.
    return int(np.digitize(value, [-.4, .65, 1.55]) + 2)


def generate(seed=42, students=360):
    rng = np.random.default_rng(seed)
    start = date(2026, 2, 2)
    grades, events = [], []
    for i in range(1, students + 1):
        student_id = f'S{i:04d}'
        group_code = f'ПИ-4{(i-1)//30+1:02d}'
        ability = rng.normal(.35, .9)
        diligence = rng.normal(.4, .8)
        for course in COURSES:
            difficulty = {'ud': .10, 'bd': .25, 'pr': .35}[course]
            level = ability - difficulty + rng.normal(0, .3)
            control_1 = grade(level + .2*diligence + rng.normal(0, .6))
            control_2 = grade(level + .3*diligence + rng.normal(0, .6))
            shock = rng.normal(0, .7) - (rng.random() < .12)*.9
            final_grade = grade(level + .4*diligence + shock)
            grades.append(dict(student_id=student_id, group_code=group_code, course=course,
                semester=8, control_1=control_1, control_2=control_2, final_grade=final_grade))
            p_attend = 1/(1+np.exp(-(1.1 + .7*diligence + .25*ability)))
            p_submit = 1/(1+np.exp(-(.8 + .8*diligence + .4*level)))
            for week in range(1, 17):
                for day in [0, 3]:
                    observed = start + timedelta(weeks=week-1, days=day)
                    events.append(dict(event_id=f'{course}-{student_id}-C{week:02d}{day}',
                        student_id=student_id, course=course, event_date=observed.isoformat(),
                        event_type='attendance', attended=int(rng.random()<p_attend), submitted=None, late=None))
                if week % 2 == 0:
                    observed = start + timedelta(weeks=week-1, days=4)
                    submitted = int(rng.random()<p_submit)
                    late = int(submitted and rng.random() < .12 + .18/(1+np.exp(diligence)))
                    # Статус уже известен на указанную дату наблюдения.
                    events.append(dict(event_id=f'{course}-{student_id}-L{week//2:02d}',
                        student_id=student_id, course=course, event_date=observed.isoformat(),
                        event_type='lab', attended=None, submitted=submitted, late=late))
    g, e = pd.DataFrame(grades), pd.DataFrame(events)
    g.loc[[7, 111, 802], 'control_2'] = np.nan
    g.loc[5, 'course'] = ' '+g.loc[5,'course']+' '
    g = pd.concat([g, g.iloc[[1, 19, 44, 103, 201]]], ignore_index=True)
    e = pd.concat([e, e.iloc[:12]], ignore_index=True)
    (DATA/'raw').mkdir(parents=True, exist_ok=True)
    g.to_csv(DATA/'raw/gradebook.csv', index=False)
    e.to_csv(DATA/'raw/activity_log.csv', index=False)
    meta = dict(is_synthetic=True, country_context='Россия', institution='Учебный институт Южный кампус',
        seed=seed, students=students, courses=COURSES, semester=8, weeks=16, as_of=AS_OF,
        scale=[2,3,4,5], generated_grade_rows=len(g), generated_activity_rows=len(e),
        data_generator='src/generate_data.py',
        disclosure='Все записи вымышлены. Вуз и его внутренние правила вымышлены. Это не данные РУДН.',
        latent_variables='Способность и организованность влияют на результаты. Итог содержит независимый шум и будущий случайный шок.',
        defects='5 дубликатов ведомости, 12 дубликатов событий, 3 пропуска control_2, пробелы в коде дисциплины.')
    (DATA/'generation_manifest.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    generate()
