# 📉 Customer Churn Risk Advisor

An interactive decision-support application built with **Streamlit** and **XGBoost** to predict customer churn probability, assign risk bands, and provide prescriptive retention strategies for telco operators.

---

## 📋 Model Card: Churn Risk Advisor v1.0

* **Intended use:** Rank existing telecom customers by churn risk so a retention team can prioritize calls. Decision support, not automation.
* **Not for:** Credit, pricing, or any decision that denies a service.
* **Data:** IBM Telco Customer Churn, 7,043 customers, 26.5% churn.
* **Model:** XGBClassifier, chosen by 5-fold CV. Features: 30 encoded columns.
* **Performance:** CV AUC 0.845 +/- 0.012; test AUC 0.842 (test set used once).
* **Threshold:** 0.35, from a cost analysis (FN = PKR 6,000, FP = PKR 1,000).
* **Limitations:** One US dataset from one period; no Pakistan data; associations, not causes; performance may drift as plans change.
* **Fairness check (Week 15):** Compare recall across gender and seniors.
* **Owner and version:** Rafi Haider, v1.0, October 9, 2026.

---

## 🚀 Live Application & Repository
- **Live Web App:** [Customer Churn Risk Advisor on Streamlit Cloud](https://churn-risk-advisor-td2kkqsa9zbfrf8scabsbk.streamlit.app/) 
- **GitHub Repository:** [RafiHaider/churn-risk-advisor](https://github.com/RafiHaider/churn-risk-advisor)
  
## 📚 Notebook References
- **Week 1 (EDA & Data Prep):** [1st Week Customer Churn EDA](https://www.kaggle.com/code/rafihaider/1st-week-customer-churn-eda)
- **Week 2 (Model Building):** [Week 2 Building, Evaluating, and Interpreting ML](https://www.kaggle.com/code/rafihaider/week-2-building-evaluating-and-interpreting-ml)
- **Week 3 (Optimization):** [Week 3 Model Optimization and Unsupervised Learning](https://www.kaggle.com/code/rafihaider/week-3-model-optimization-and-unsupervised-learn)

---

## 📌 Project Overview & Features

The **Churn Risk Advisor** bridges the gap between raw machine learning output and real-world business intervention[cite: 1]. Key capabilities include:

- **Real-time Individual Scoring:** Input specific customer attributes via a sidebar form to instantaneously generate a churn probability score, assigned risk band (LOW / MEDIUM / HIGH), and recommended outreach action.
  
  <img width="953" height="502" alt="Screenshot 2026-10-09 092359(1)" src="https://github.com/user-attachments/assets/be696a4d-f43f-4899-93be-09b6005c8f12" />

- **Prescriptive Counterfactual Analysis:** Dynamic *"What would change the risk?"* feature computes probability deltas for potential contract and service upgrades.

- **Batch Scoring & Export:** Drag and drop CSV customer data to process batch predictions, highlight high-risk accounts above a customizable threshold, and export scored reports.

  <img width="944" height="477" alt="Screenshot 2026-10-09 093234" src="https://github.com/user-attachments/assets/7f012787-1a2c-46b2-a92b-f0602a577a19" />
  
  <img width="941" height="478" alt="Screenshot 2026-10-09 103725" src="https://github.com/user-attachments/assets/ae70db98-290f-432d-ac40-57570583fe9b" />

- **Explainable AI (XAI):** Feature contribution breakdowns showing key factors driving risk up or down.

---

## 📊 Model Performance & Specification

- **Algorithm:** `XGBClassifier`
- **Validation Performance:** Cross-Validation AUC of **$0.845 \pm 0.012$**
- **Feature Schema:** 19 customer attributes including tenure, contract type, payment method, internet services, and billing preferences.

---

## 💻 Local Setup & Execution

Run the following commands in your terminal to set up and launch the application locally:

```bash
# 1. Clone the repository
git clone [https://github.com/RafiHaider/churn-risk-advisor.git](https://github.com/RafiHaider/churn-risk-advisor.git)
cd churn-risk-advisor

# 2. Create and activate a virtual environment
python -m venv .venv

# On Windows:
.venv\Scripts\activate

# On macOS/Linux:
source .venv/bin/activate

# 3. Install required dependencies
pip install -r requirements.txt

# 4. Launch the Streamlit application
streamlit run app.py
