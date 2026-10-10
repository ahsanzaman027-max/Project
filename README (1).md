# Churn Risk Advisor v1.0

A Streamlit app that scores telecom customers for churn risk and tells a retention team who to call first.

**Live app:** [PASTE YOUR STREAMLIT URL HERE](https://your-app-name.streamlit.app)

---

## Model card: Churn Risk Advisor v1.0

**Intended use:** rank existing telecom customers by churn risk so a retention team can prioritize calls. Decision support, not automation.

**Not for:** credit, pricing, or any decision that denies a service.

**Data:** IBM Telco Customer Churn, 7,043 customers, 26.5% churn.

**Model:** XGBoost (tuned), chosen by 5-fold CV. Features: 30 encoded columns.

**Performance:** CV AUC 0.850 +/- 0.012; test AUC 0.848 (test set used once). At the 0.5 default cut-off the test recall is 0.521 and precision 0.659.

**Threshold:** 0.30, from a cost analysis (FN = PKR 6,000, FP = PKR 1,000). The app uses three bands: LOW below 15%, WATCH 15% to 30%, HIGH from 30% (contact now).

**Limitations:** one US dataset from one period; no Pakistan data; associations, not causes; performance may drift as plans change.

**Fairness check (Week 15):** compare recall across gender and seniors.

**Owner and version:** Muhammad Umar Khan, v1.0, 10 October 2026.

---

## Screenshots

### Single customer: score, risk band, what-if and explanation
The sidebar describes one customer. The app shows the churn probability, the risk band, the recommended action, and a what-if chart showing how the risk changes under alternative contract, payment or support options.

![Single customer view](images/single_customer.png)

### Batch scoring: upload a customer file
Uploading `WA_Fn-UseC_-Churn.csv` scores all 7,043 customers: 2,692 (38%) are above the contact threshold and the average risk is 27%. The table is sorted by churn probability so the team can start at the top.

![Batch scoring view](images/batch_scoring.png)

---

## Run locally

```bash
# 1. get the code
git clone <YOUR-REPO-URL>
cd <YOUR-REPO-FOLDER>

# 2. create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. install the pinned dependencies (scikit-learn==1.6.1, the training version)
pip install -r requirements.txt

# 4. start the app
streamlit run app.py
```

The app opens at http://localhost:8501. Required files in the repo: `app.py`, `churn_model.joblib`, `model_meta.json`, `sample_customers.csv`, `requirements.txt`.

---

## Notebooks

- [Week 1 notebook](PASTE-LINK)
- [Week 2 notebook](PASTE-LINK)
- [Week 3 notebook: model optimization](PASTE-LINK)
- [Lab 4 notebook: packaging and parity test](PASTE-LINK)
