from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
ARTIFACTS = ROOT / 'artifacts'
DB_PATH = DATA / 'education.sqlite'
SEED = 42
AS_OF = '2026-03-29'
COURSES = {'ud': 'Управление данными', 'bd': 'Базы данных', 'pr': 'Программирование'}
NUMERIC_FEATURES = ['control_1', 'control_2', 'attendance_rate', 'lab_completion', 'late_share']
CATEGORICAL_FEATURES = ['course']
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def add_features(frame):
    frame = frame.copy()
    frame['attendance_rate'] = frame['classes_attended'] / frame['classes_total'].replace(0, float('nan'))
    frame['lab_completion'] = frame['labs_submitted'] / frame['labs_assigned'].replace(0, float('nan'))
    frame['late_share'] = frame['late_labs'] / frame['labs_assigned'].replace(0, float('nan'))
    return frame
