# Banking Customer Intelligence & Churn Analytics

An interactive Streamlit slide deck that takes a bank churn dataset through the full data science flow:
cleaning, exploratory analysis, segmentation, feature engineering, model comparison, evaluation,
profit-based targeting and customer risk scoring.

## Slides
1. Executive summary
2. Data & quality
3. Exploratory analysis
4. Customer segments
5. Feature engineering
6. Model comparison (Logistic Regression, KNN, Random Forest, Gradient Boosting)
7. Evaluation & profit (retention budget simulator)
8. Risk scoring (live single-customer prediction)
9. Decisions & actions (risk tiers and top customers to contact)

## Run locally
```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

## Project structure
```
app.py             # the whole app
churn.csv          # dataset (10,000 bank customers)
requirements.txt
.streamlit/config.toml
```

## Tech
Python, Streamlit, pandas, scikit-learn, Plotly.
