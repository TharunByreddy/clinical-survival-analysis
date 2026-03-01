# Results Directory

## Output Files
All plots and outputs are generated after running the pipeline:

| File | Description |
|------|-------------|
| `kaplan_meier_curves.png` | KM survival curves by risk group and treatment |
| `hazard_ratios.png` | Cox PH hazard ratio bar chart |
| `eda_distributions.png` | EDA distribution plots |
| `roc_curve.png` | ROC curve for classification |
| `shap_summary.png` | SHAP feature importance |

## How to generate
```bash
# Run the notebook
jupyter notebook notebooks/survival_analysis.ipynb

# Or run the Streamlit dashboard
streamlit run app/app.py
```

## Key Results
| Metric | Value |
|--------|-------|
| Concordance Index | 0.74 |
| Log-rank p-value | < 0.001 |
| Patients Analyzed | 500+ |
