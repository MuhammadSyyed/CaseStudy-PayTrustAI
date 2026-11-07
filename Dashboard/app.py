from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="UniFee Guard — Governance Dashboard", layout="wide")

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "Dataset" / "unifee_admission_fees.csv"

GOVERNED_FEATURES = [
    "fee_amount_pkr",
    "campus",
    "program",
    "payment_channel",
    "hour",
    "new_device",
    "payment_velocity_24h",
    "geo_anomaly_score",
    "spending_deviation_score",
]
LEAKAGE_OR_ID = ["payment_id", "student_id"]
POLICY = {"max_fpr": 0.08, "min_recall": 0.55, "min_f1": 0.40, "max_campus_fpr_gap": 0.08}


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


def encode_frame(df, columns):
    out = df.copy()
    for col in columns:
        if not pd.api.types.is_numeric_dtype(out[col]):
            out[col] = LabelEncoder().fit_transform(out[col].astype(str))
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)
    return out


def fpr_score(y_true, y_pred):
    denom = max((y_true == 0).sum(), 1)
    return ((y_pred == 1) & (y_true == 0)).sum() / denom


def evaluate(name, model, X_train, X_test, y_train, y_test, scale=False):
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1-Score": f1_score(y_test, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, prob),
        "FPR": fpr_score(y_test.values, pred),
        "_pred": pred,
        "_y": y_test.values,
        "_campus": None,
    }


@st.cache_data
def run_models(df):
    rng_state = 42
    results = {"without": [], "with": []}

    # WITHOUT governance: keep IDs, accuracy-oriented feel via class_weight only on trees
    cols_u = LEAKAGE_OR_ID + GOVERNED_FEATURES
    data_u = encode_frame(df[cols_u + ["is_fraud"]], cols_u)
    X_u = data_u.drop(columns=["is_fraud"])
    y = data_u["is_fraud"]
    Xu_tr, Xu_te, yu_tr, yu_te = train_test_split(
        X_u, y, test_size=0.25, random_state=rng_state, stratify=y
    )
    models_u = {
        "Logistic Regression": (LogisticRegression(max_iter=1000, random_state=rng_state), True),
        "Decision Tree": (DecisionTreeClassifier(max_depth=12, random_state=rng_state), False),
        "Random Forest": (
            RandomForestClassifier(n_estimators=120, max_depth=14, random_state=rng_state, n_jobs=-1),
            False,
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=rng_state),
            False,
        ),
    }
    for name, (model, scale) in models_u.items():
        row = evaluate(name, model, Xu_tr, Xu_te, yu_tr, yu_te, scale=scale)
        results["without"].append(row)

    # WITH governance: approved features + class_weight balanced
    data_g = encode_frame(df[GOVERNED_FEATURES + ["is_fraud"]].copy(), GOVERNED_FEATURES)
    campus_raw = df["campus"].copy()
    X_g = data_g[GOVERNED_FEATURES]
    y_g = data_g["is_fraud"]
    Xg_tr, Xg_te, yg_tr, yg_te, c_tr, c_te = train_test_split(
        X_g, y_g, campus_raw, test_size=0.25, random_state=rng_state, stratify=y_g
    )
    models_g = {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=rng_state),
            True,
        ),
        "Decision Tree": (
            DecisionTreeClassifier(max_depth=10, class_weight="balanced", random_state=rng_state),
            False,
        ),
        "Random Forest": (
            RandomForestClassifier(
                n_estimators=120,
                max_depth=12,
                class_weight="balanced",
                random_state=rng_state,
                n_jobs=-1,
            ),
            False,
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=rng_state),
            False,
        ),
    }
    for name, (model, scale) in models_g.items():
        row = evaluate(name, model, Xg_tr, Xg_te, yg_tr, yg_te, scale=scale)
        row["_campus"] = c_te.values
        results["with"].append(row)

    def to_df(rows):
        return pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])

    return to_df(results["without"]), to_df(results["with"]), results


def campus_fpr_gap(row):
    if row.get("_campus") is None:
        return np.nan
    y, pred, campus = row["_y"], row["_pred"], row["_campus"]
    gaps = []
    for c in np.unique(campus):
        mask = campus == c
        gaps.append(fpr_score(y[mask], pred[mask]))
    return float(np.max(gaps) - np.min(gaps)) if gaps else np.nan


def main():
    st.title("UniFee Guard")
    st.caption(
        "Exercise 4 dashboard — University Admission Fees fraud governance "
        "(separate organisation from PayTrust AI)"
    )

    df = load_data()
    without_df, with_df, raw = run_models(df)

    tabs = st.tabs(
        [
            "Case overview",
            "Dataset governance form",
            "Model form",
            "ML results",
            "Risk register",
            "Final decision",
        ]
    )

    with tabs[0]:
        st.subheader("Organisation case study")
        st.markdown(
            """
**UniFee Guard** is the admissions office AI review for a multi-campus university.
The product team wants to auto-block suspicious **admission fee payments**.
A headline accuracy number looks fine, but blocked legitimate fees delay enrolment
(especially for Online-campus applicants).

This dashboard mirrors the PayTrust lab method on a **new domain**: forms, NIST-style
gates, model comparison with/without governance, and a GO / NO-GO memo.
"""
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Fee payments", f"{len(df):,}")
        c2.metric("Fraud rate", f"{df['is_fraud'].mean():.2%}")
        c3.metric("Campuses", df["campus"].nunique())
        c4.metric("Channels", df["payment_channel"].nunique())

        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
        df["is_fraud"].value_counts().plot(kind="bar", ax=axes[0], color=["#4C78A8", "#E45756"])
        axes[0].set_title("Class imbalance")
        axes[0].set_xticklabels(["Legit (0)", "Fraud (1)"], rotation=0)
        df.groupby("campus")["is_fraud"].mean().plot(kind="bar", ax=axes[1], color="#72B7B2")
        axes[1].set_title("Fraud rate by campus")
        axes[1].tick_params(axis="x", rotation=0)
        st.pyplot(fig)
        st.dataframe(df.head(10), use_container_width=True)

    with tabs[1]:
        st.subheader("Dataset Governance Form (digitized)")
        data_form = pd.DataFrame(
            [
                ("Dataset Name", "UniFee Admission Fee Payments"),
                ("Dataset ID", "UF-DS-FEES-001"),
                ("Description", "Synthetic university admission fee payments with fraud labels"),
                ("Business Purpose", "Detect fraudulent fee payments without blocking legitimate enrolment"),
                ("Source", "Generated instructional dataset for UniFee Guard case"),
                ("Format / Location", "CSV / Dataset/unifee_admission_fees.csv"),
                ("Business Owner", "Director of Admissions"),
                ("Data Steward", "University AI Governance Lead"),
                ("Classification", "Confidential"),
                ("Personal / Sensitive Data", "Yes - student_id (restricted)"),
                ("Quality risks", "Class imbalance; campus mix shift during intake peaks"),
                ("Permitted Use", "Fraud scoring evaluation and governance demonstration"),
                ("Governance Status", "Under Review"),
                ("Approval Date", str(date.today())),
            ],
            columns=["Field", "Value"],
        )
        st.dataframe(data_form, use_container_width=True, hide_index=True)

    with tabs[2]:
        st.subheader("Model Documentation Form (digitized)")
        model_form = pd.DataFrame(
            [
                ("Model name", "UniFee-FeeFraud-RF-v1"),
                ("Provider", "University Admissions Analytics (instructional)"),
                ("Intended uses", "Risk-score admission fee payments; human review for mid band"),
                ("Prohibited uses", "Academic grading; scholarship decisions; ID verification alone"),
                ("Approved features", ", ".join(GOVERNED_FEATURES)),
                ("Excluded features", ", ".join(LEAKAGE_OR_ID)),
                ("Primary metrics", "Precision, Recall, F1, ROC-AUC, FPR, campus FPR gap"),
                ("Human oversight", "Admissions officers review uncertain scores before hard block"),
                ("Monitoring and rollback", "Weekly campus FPR monitoring; revert to manual checks"),
                ("Model owner", "Admissions Analytics Lead"),
                ("Governance Status", "Under Review"),
            ],
            columns=["Field", "Value"],
        )
        st.dataframe(model_form, use_container_width=True, hide_index=True)

    with tabs[3]:
        st.subheader("ML results — with vs without governance")
        left, right = st.columns(2)
        left.markdown("**Without governance** (IDs allowed, no class_weight policy)")
        left.dataframe(without_df.style.format("{:.3f}", subset=without_df.columns[1:]), use_container_width=True)
        right.markdown("**With governance** (approved features + balanced class weight)")
        right.dataframe(with_df.style.format("{:.3f}", subset=with_df.columns[1:]), use_container_width=True)

        best_u = without_df.sort_values("F1-Score", ascending=False).iloc[0]
        best_g = with_df.sort_values("F1-Score", ascending=False).iloc[0]
        fig, ax = plt.subplots(figsize=(9, 4))
        metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "FPR"]
        x = np.arange(len(metrics))
        ax.bar(x - 0.18, [best_u[m] for m in metrics], width=0.36, label=f"Without ({best_u['Model']})")
        ax.bar(x + 0.18, [best_g[m] for m in metrics], width=0.36, label=f"With ({best_g['Model']})")
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=20)
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.set_title("Best model per pipeline")
        st.pyplot(fig)

        # blocked legit vs missed fraud for best governed model
        best_g_raw = max(raw["with"], key=lambda r: r["F1-Score"])
        y, pred = best_g_raw["_y"], best_g_raw["_pred"]
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        st.info(f"Governed best model blocked **{fp}** legitimate fee payments and missed **{fn}** fraudulent ones on the hold-out set.")

    with tabs[4]:
        st.subheader("Risk register")
        risk = pd.DataFrame(
            [
                ("R1", "Legitimate enrolment fees blocked", 5, 4, "Admissions Lead", "Human review queue; raise precision floor"),
                ("R2", "Campus disparity in false blocks", 4, 3, "Governance Lead", "Monitor campus FPR gap; recalibrate"),
                ("R3", "ID leakage into model features", 5, 2, "ML Lead", "Enforce approved feature list from Model form"),
                ("R4", "Intake-season drift", 4, 4, "Analytics Lead", "Weekly monitoring + rollback to manual"),
                ("R5", "Incomplete appeals path", 5, 3, "Registrar", "Publish appeal SLA before full auto-block"),
            ],
            columns=["ID", "Risk", "Impact (1-5)", "Likelihood (1-5)", "Owner", "Mitigation"],
        )
        risk["Score"] = risk["Impact (1-5)"] * risk["Likelihood (1-5)"]
        st.dataframe(risk.sort_values("Score", ascending=False), use_container_width=True, hide_index=True)

    with tabs[5]:
        st.subheader("Final deployment decision")
        best_u = without_df.sort_values("F1-Score", ascending=False).iloc[0]
        best_g = with_df.sort_values("F1-Score", ascending=False).iloc[0]
        best_g_raw = max(raw["with"], key=lambda r: r["F1-Score"])
        gap = campus_fpr_gap(best_g_raw)

        gates = pd.DataFrame(
            [
                ("Dataset form complete", True),
                ("Model form complete", True),
                ("IDs excluded under governance", True),
                (f"Governed FPR <= {POLICY['max_fpr']:.0%}", best_g["FPR"] <= POLICY["max_fpr"]),
                (f"Governed Recall >= {POLICY['min_recall']:.0%}", best_g["Recall"] >= POLICY["min_recall"]),
                (f"Governed F1 >= {POLICY['min_f1']:.0%}", best_g["F1-Score"] >= POLICY["min_f1"]),
                (
                    f"Campus FPR gap <= {POLICY['max_campus_fpr_gap']:.0%}",
                    (gap <= POLICY["max_campus_fpr_gap"]) if pd.notna(gap) else False,
                ),
            ],
            columns=["Gate", "Pass"],
        )
        st.dataframe(gates, use_container_width=True, hide_index=True)

        all_pass = bool(gates["Pass"].all())
        if all_pass:
            decision = "CONDITIONAL GO — controlled pilot"
            color = "green"
            next_step = "Pilot on one campus with human review; expand only if campus FPR gap stays inside policy."
        elif best_g["F1-Score"] >= best_u["F1-Score"] * 0.9:
            decision = "NO-GO for full auto-block / SHADOW MODE"
            color = "orange"
            next_step = "Keep scoring visible to officers only; remediate failed Measure gates before auto-blocking fees."
        else:
            decision = "NO-GO"
            color = "red"
            next_step = "Halt auto decisions; rework features, imbalance handling, and appeals."

        st.markdown(f"### Decision: :{color}[{decision}]")
        st.markdown(
            f"""
**CTO memo — UniFee Guard ({date.today()})**

- Without-governance best model: **{best_u['Model']}** (F1={best_u['F1-Score']:.3f}, FPR={best_u['FPR']:.3f})
- With-governance best model: **{best_g['Model']}** (F1={best_g['F1-Score']:.3f}, FPR={best_g['FPR']:.3f}, campus FPR gap={gap:.3f})
- Forms integrated: Dataset Governance Form + Model Documentation Form
- Next step: {next_step}

A strong accuracy story is not enough when a false block delays a student's admission fee.
Governance (approved features, imbalance handling, campus fairness, human review) is required before launch.
"""
        )


if __name__ == "__main__":
    main()
