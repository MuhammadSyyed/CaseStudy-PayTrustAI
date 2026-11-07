# PayTrust AI + UniFee Guard — AI Governance Lab Submission

Submission package for the PayTrust AI case study exercises (`Documents/Execises.txt`).

## Exercise map

| Exercise | Deliverable | What it shows |
|----------|-------------|---------------|
| **1** | [`Notebooks/01_Governance_Forms_Pipeline.ipynb`](Notebooks/01_Governance_Forms_Pipeline.ipynb) | Digitized **Dataset Governance Form** + **Model Documentation Form** wired into NIST Govern/Map/Measure/Manage gates → GO / CONDITIONAL / NO-GO |
| **2** | [`Notebooks/02_Fraud_ML_WithWithout_Governance.ipynb`](Notebooks/02_Fraud_ML_WithWithout_Governance.ipynb) (first half) | Online fraud dataset registered in the Data Governance Form and loaded into the pipeline |
| **3** | Same notebook `02_...` (models section) | 4 ML models + class imbalance (`class_weight`, SMOTE/RUS) compared **with vs without** governance |
| **4** | [`Dashboard/app.py`](Dashboard/app.py) | **UniFee Guard** Streamlit dashboard — university admission fees (new organisation, not PayTrust) |

Supporting references (kept as-is):

- [`Notebooks/PayTrust_AI_Governance.ipynb`](Notebooks/PayTrust_AI_Governance.ipynb) — original lab Parts 1–15  
- [`Notebooks/fraud_detection_model.ipynb`](Notebooks/fraud_detection_model.ipynb) — base fraud ML exploration  
- [`Governance Forms/`](Governance%20Forms/) — PDF forms used as field templates  
- [`Dataset/financial_fraud_detection_dataset.csv`](Dataset/financial_fraud_detection_dataset.csv) — online fraud data (Ex 2–3)  
- [`Dataset/unifee_admission_fees.csv`](Dataset/unifee_admission_fees.csv) — synthetic university fee payments (Ex 4)

## How to run notebooks

1. Open the project in VS Code / Cursor / Jupyter / Google Colab.
2. Run cells top-to-bottom:
   - `Notebooks/01_Governance_Forms_Pipeline.ipynb`
   - `Notebooks/02_Fraud_ML_WithWithout_Governance.ipynb`
3. For notebook `02`, optional but recommended:

```bash
pip install imbalanced-learn
```

Without `imbalanced-learn`, the notebook still runs using `class_weight='balanced'`.

**Note:** The online fraud CSV has 5M rows. Notebook `02` samples all fraud rows plus a non-fraud subset so training stays practical.

## How to run the UniFee Guard dashboard (Exercise 4)

```bash
cd "Dashboard"
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Tabs: Case overview → Dataset form → Model form → ML results → Risk register → Final decision.

## CTO decisions (summary)

**PayTrust AI (Exercises 1–3):** Do not approve an unrestricted live launch based on “96% accuracy” alone. Integrate Data/Model forms into the NIST pipeline, train on the registered online fraud dataset with imbalance handling, and prefer the **with-governance** setup (no ID/leakage features, FPR/recall gates, human review / shadow mode until Measure gates pass).

**UniFee Guard (Exercise 4):** Same governance method on a **university admission fees** case. Auto-blocking fee payments can delay enrolment; require completed forms, approved features, campus fairness checks, and a controlled pilot or shadow mode before full automation.