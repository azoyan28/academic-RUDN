"""EDA, групповые выборки, подбор моделей и единственная итоговая оценка."""
import json
import sqlite3
from datetime import datetime, timezone

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score, RocCurveDisplay,
)
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import ARTIFACTS, CATEGORICAL_FEATURES, DATA, DB_PATH, FEATURES, NUMERIC_FEATURES, SEED, COURSES
from src.etl import run_etl


def make_preprocessor():
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), NUMERIC_FEATURES),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
         CATEGORICAL_FEATURES),
    ])


def split_groups(frame):
    idx = np.arange(len(frame))
    trainval, test = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=SEED).split(
        idx, frame.risk_target, frame.entity_key))
    ti, vi = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=SEED + 1).split(
        trainval, frame.iloc[trainval].risk_target, frame.iloc[trainval].entity_key))
    return {"train": trainval[ti], "validation": trainval[vi], "test": test}


def metrics(y, p, threshold=.5):
    predicted = p >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {"n": len(y), "positives": int(np.sum(y)), "threshold": float(threshold),
            "accuracy": float(accuracy_score(y, predicted)),
            "precision": float(precision_score(y, predicted, zero_division=0)),
            "recall": float(recall_score(y, predicted, zero_division=0)),
            "f1": float(f1_score(y, predicted, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, p)),
            "average_precision": float(average_precision_score(y, p)),
            "brier": float(brier_score_loss(y, p)),
            "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]]}


def save_eda(frame):
    figdir = ARTIFACTS / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
        "axes.spines.top": False,"axes.spines.right":False,"figure.dpi":160})
    fig,ax=plt.subplots(figsize=(8,4))
    grades=np.arange(2,6)
    for i,(course,color) in enumerate(zip(COURSES,["#326781","#bd7535","#7c929f"])):
        counts=frame.loc[frame.course.eq(course),"final_grade"].value_counts().reindex(grades,fill_value=0)
        ax.bar(grades+(i-1)*.23,counts,width=.23,label=COURSES[course],color=color)
    ax.set_xticks(grades)
    ax.set(xlabel="Итоговая оценка по шкале 2–5",ylabel="Число записей",title="Синтетическое распределение оценок")
    ax.legend(fontsize=9);fig.tight_layout();fig.savefig(figdir/"grades.png");plt.close(fig)
    cols=["control_1","control_2","final_grade","attendance_rate","lab_completion","late_share"]
    labels=["Контрольная 1","Контрольная 2","Итог","Посещаемость","Доля сдачи","Просрочки"]
    corr=frame[cols].corr()
    fig,ax=plt.subplots(figsize=(7,5))
    im=ax.imshow(corr,cmap="RdBu_r",vmin=-1,vmax=1)
    ax.set_xticks(range(len(cols)),labels,rotation=35,ha="right",fontsize=9)
    ax.set_yticks(range(len(cols)),labels,fontsize=9)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j,i,f"{corr.iloc[i,j]:.2f}",ha="center",va="center",fontsize=9,
                    color="white" if abs(corr.iloc[i,j])>.7 else "black")
    fig.colorbar(im,ax=ax,label="Корреляция Пирсона");fig.tight_layout()
    fig.savefig(figdir/"correlations.png");plt.close(fig)
    stats=frame.groupby("course").agg(records=("record_id","size"),mean_grade=("final_grade","mean"),
        sd_grade=("final_grade","std"),median_grade=("final_grade","median"),
        risk_count=("risk_target","sum"),risk_share=("risk_target","mean"),
        mean_attendance=("attendance_rate","mean"),mean_lab_completion=("lab_completion","mean"))
    stats.to_csv(ARTIFACTS/"course_statistics.csv")
    frame.select_dtypes("number").describe().to_csv(ARTIFACTS/"descriptive_statistics.csv")
    corr.to_csv(ARTIFACTS/"correlations.csv")
    return stats.reset_index().to_dict("records")


def train():
    ARTIFACTS.mkdir(exist_ok=True)
    frame, quality = run_etl()
    course_stats = save_eda(frame)
    splits = split_groups(frame)
    sets = {name: set(frame.iloc[idx].entity_key) for name, idx in splits.items()}
    assert not (sets["train"] & sets["validation"] or sets["train"] & sets["test"] or sets["validation"] & sets["test"])
    assignment = frame[["record_id", "entity_key", "risk_target"]].copy()
    for name, idx in splits.items():
        assignment.loc[idx, "split"] = name
    assignment.to_csv(ARTIFACTS / "splits.csv", index=False)
    X, y = frame[FEATURES], frame.risk_target
    tr, va, te = splits["train"], splits["validation"], splits["test"]
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_report, tuned = [], {}
    models = [
        ("logistic", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED),
         {"model__C": [.1, 1., 10.]}),
        ("random_forest", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=SEED, n_jobs=1),
         {"model__max_depth": [4, 8, None], "model__min_samples_leaf": [2, 5]}),
    ]
    for name, model, grid in models:
        pipe = Pipeline([("prep", make_preprocessor()), ("model", model)])
        search = GridSearchCV(pipe, grid, scoring="average_precision", cv=cv, n_jobs=1, refit=True)
        search.fit(X.iloc[tr], y.iloc[tr], groups=frame.iloc[tr].entity_key)
        tuned[name] = search.best_estimator_
        cv_report.append({"name": name, "best_params": search.best_params_,
                          "cv_average_precision": float(search.best_score_),
                          "validation": metrics(y.iloc[va], search.predict_proba(X.iloc[va])[:, 1])})
    # Модель выбирается только по CV на обучающей части.
    best = max(cv_report, key=lambda r: r["cv_average_precision"])
    selected = tuned[best["name"]]
    vp = selected.predict_proba(X.iloc[va])[:, 1]
    thresholds = np.arange(.1, .901, .025)
    scores = [f1_score(y.iloc[va], vp >= t, zero_division=0) for t in thresholds]
    # При одинаковом F1 выбираем больший порог, уменьшая число сигналов.
    threshold = float(max(t for t, s in zip(thresholds, scores) if s == max(scores)))
    baseline = DummyClassifier(strategy="prior").fit(X.iloc[tr], y.iloc[tr])
    test_results = {"dummy_prior": metrics(y.iloc[te], baseline.predict_proba(X.iloc[te])[:, 1])}
    # К этому моменту архитектура, параметры и порог зафиксированы.
    for name, model in tuned.items():
        test_results[name] = metrics(y.iloc[te], model.predict_proba(X.iloc[te])[:, 1], threshold if name == best["name"] else .5)
    probs = selected.predict_proba(X.iloc[te])[:, 1]
    output = frame.iloc[te][["record_id", "entity_key", "course", "final_grade", "risk_target"]].copy()
    output["risk_score"] = probs
    output["prediction"] = (probs >= threshold).astype(int)
    output.to_csv(ARTIFACTS / "test_predictions.csv", index=False)
    # Интерпретация оценивается на валидации, а не по обучающим весам.
    importance = permutation_importance(selected, X.iloc[va], y.iloc[va], n_repeats=10,
                                         scoring="average_precision", random_state=SEED)
    fi = pd.DataFrame({"feature": FEATURES, "importance": importance.importances_mean,
                       "sd": importance.importances_std}).sort_values("importance", ascending=False)
    fi.to_csv(ARTIFACTS / "feature_importance.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    top = fi.head(8).iloc[::-1]
    ax.barh(top.feature, top.importance, xerr=top.sd, color="#326781")
    ax.set(xlabel="Снижение Average Precision при перестановке", title="Важность признаков на валидации")
    fig.tight_layout(); fig.savefig(ARTIFACTS / "figures" / "importance.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, model in tuned.items():
        RocCurveDisplay.from_predictions(y.iloc[te], model.predict_proba(X.iloc[te])[:,1], name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set(title="ROC на отложенной тестовой выборке")
    fig.tight_layout(); fig.savefig(ARTIFACTS / "figures" / "roc.png"); plt.close(fig)
    cm = test_results[best["name"]]["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):ax.text(j, i, str(cm[i][j]), ha="center", va="center", fontsize=20,
                                 color="white" if cm[i][j] > np.max(cm)*.5 else "black")
    ax.set_xticks([0, 1], ["Низкий риск", "Риск"]); ax.set_yticks([0, 1], ["Итог 3–5", "Итог 2"])
    ax.set(xlabel="Прогноз", ylabel="Факт", title="Матрица ошибок выбранной модели")
    fig.tight_layout(); fig.savefig(ARTIFACTS / "figures" / "confusion.png"); plt.close(fig)
    # Групповой bootstrap сохраняет зависимость записей одного профиля.
    rng, boot = np.random.default_rng(SEED), []
    test = frame.iloc[te].reset_index(drop=True)
    group_rows = {k: np.flatnonzero(test.entity_key.to_numpy() == k) for k in test.entity_key.unique()}
    keys = list(group_rows)
    for _ in range(500):
        sampled = rng.choice(keys, size=len(keys), replace=True)
        ix = np.concatenate([group_rows[k] for k in sampled])
        if test.iloc[ix].risk_target.nunique() > 1:
            boot.append(f1_score(test.iloc[ix].risk_target, probs[ix] >= threshold, zero_division=0))
    report = {"seed": SEED, "quality": quality, "features": FEATURES, "excluded": ["final_grade", "risk_target", "events_after_as_of", "generator_latent_variables"],
              "split": {k: {"records": len(v), "groups": len(sets[k]),
                        "risk_count": int(y.iloc[v].sum()), "risk_share": float(y.iloc[v].mean())}
                        for k, v in splits.items()},
              "cv": cv_report, "selected_model": best["name"], "threshold": threshold,
              "validation_selected": metrics(y.iloc[va], vp, threshold),
              "test": test_results, "f1_group_bootstrap_95ci": np.quantile(boot, [.025, .975]).tolist(),
              "course_statistics": course_stats,
              "feature_importance": fi.to_dict("records")}
    (ARTIFACTS / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    joblib.dump({"pipeline": selected, "threshold": threshold, "features": FEATURES,
                 "selected_model": best["name"]}, ARTIFACTS / "model.joblib")
    with sqlite3.connect(DB_PATH) as con:
        for name, result in test_results.items():
            con.execute("INSERT INTO experiments(created_at,model_name,split_name,metrics_json) VALUES (?,?,?,?)",
                        (datetime.now(timezone.utc).isoformat(), name, "test", json.dumps(result)))
    print(json.dumps({"selected_model": best["name"], "threshold": threshold,
                      "split": report["split"], "test": test_results}, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    train()
