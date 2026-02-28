# 🏥 Clinical Survival Analysis Dashboard

![Python](https://img.shields.io/badge/Python-3.9-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![Lifelines](https://img.shields.io/badge/Lifelines-Survival%20Analysis-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

## 📌 Overview
End-to-end clinical survival analysis project predicting patient outcomes
using Cox Proportional Hazards modeling, with an interactive Streamlit
dashboard for real-time exploration and visualization.

Inspired by real-world transplant outcome research conducted at
**Stanford University School of Medicine** on 500K+ patient records.

---

## 🎯 Problem Statement
Predicting time-to-event outcomes (survival, readmission, transplant success)
is critical in clinical research. This project builds, evaluates, and
visualizes survival models to identify key patient risk factors and support
clinical decision-making.

---

## 📁 Project Structure
```
clinical-survival-analysis/
├── app/
│   └── app.py                  # Streamlit dashboard
├── notebooks/
│   └── survival_analysis.ipynb # Full analysis walkthrough
├── src/
│   └── preprocess.py           # Data preprocessing pipeline
├── data/                       # Dataset directory
├── results/                    # Output plots and metrics
├── requirements.txt            # Dependencies
└── README.md
```

---

## 🔬 Approach
1. Synthetic clinical dataset generation (500+ patients)
2. Exploratory Data Analysis (EDA)
3. Kaplan-Meier survival curves by risk group and treatment
4. Log-rank statistical significance testing
5. Cox Proportional Hazards model training
6. Hazard ratio interpretation and feature importance
7. Interactive Streamlit dashboard deployment

---

## 📊 Results

| Metric | Score |
|--------|-------|
| Concordance Index (C-Index) | 0.74 |
| Log-rank p-value | < 0.001 |
| Patients Analyzed | 500+ |
| Features Used | 7 |

### Key Risk Factors (Hazard Ratios)
| Feature | Hazard Ratio | Interpretation |
|---------|-------------|----------------|
| Age | 1.82 | Higher age = higher risk |
| Comorbidities | 1.54 | More conditions = higher risk |
| Treatment 1 | 0.63 | Protective effect |
| Treatment 2 | 0.71 | Moderate protective effect |

---

## 🖥️ Dashboard Features
- **KPI Cards** — Total patients, events, median age, avg duration
- **Kaplan-Meier Curves** — Survival probability by risk group
- **Cox PH Model Summary** — Coefficients, hazard ratios, p-values
- **Hazard Ratio Plot** — Visual feature importance
- **Interactive Filters** — Filter by gender, diagnosis, treatment
- **Patient Data Table** — Raw data exploration

---

## 🛠️ Tech Stack
| Category | Tools |
|----------|-------|
| Language | Python 3.9 |
| Survival Analysis | Lifelines |
| ML | Scikit-learn |
| Dashboard | Streamlit |
| Visualization | Matplotlib, Plotly, Seaborn |
| Data | Pandas, NumPy |

---

## 🚀 How to Run
```bash
# 1. Clone the repo
git clone https://github.com/TharunByreddy/clinical-survival-analysis.git
cd clinical-survival-analysis

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the dashboard
streamlit run app/app.py

# 4. Or run the notebook
jupyter notebook notebooks/survival_analysis.ipynb
```

---

## 📬 Author
**Tharun Kumar Reddy Byreddy**
M.S. Statistical Data Science | San Francisco State University
[LinkedIn](https://www.linkedin.com/in/tharun-kumar-reddy-byeddy-801290215/) |
[GitHub](https://github.com/TharunByreddy)
