"""
Banking Customer Intelligence & Churn Analytics
Run:  streamlit run app.py
My data-science flow of my project: load -> clean -> EDA -> segments -> feature engineering
-> model comparison -> evaluation -> business (profit) optimisation -> scoring.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TITLE = "Banking Customer Intelligence & Churn Analytics"
BG, SKY, ORANGE, GREEN, CARD = "#0F172A", "#38BDF8", "#F97316", "#22C55E", "#E0E3FE"

st.set_page_config(page_title=TITLE, page_icon="🏦", layout="wide")

#styling here got allittle help from AI
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
@property --w {syntax:'<integer>'; initial-value:0; inherits:false}
@property --f {syntax:'<integer>'; initial-value:0; inherits:false}
html, body, .stApp, [class*="css"] {font-family:'Inter',sans-serif;}
.stApp {background:#0F172A;}
header[data-testid="stHeader"] {background:transparent;}
.block-container {padding-top:2rem; max-width:1250px;}
h1,h2,h3,h4 {color:#38BDF8 !important; letter-spacing:-0.02em;}
.stApp p, .stApp label, .stApp span, .stApp li {color:#E2E8F0;}
section[data-testid="stSidebar"] {background:#0B1220; border-right:1px solid #1E293B;}

/* staggered fade-in + slide-up for every element on a slide */
@keyframes rise {from{opacity:0; transform:translateY(18px)} to{opacity:1; transform:none}}
.block-container [data-testid="stElementContainer"] {animation:rise .6s cubic-bezier(.2,.7,.2,1) both;}
.block-container [data-testid="stElementContainer"]:nth-child(1){animation-delay:.00s}
.block-container [data-testid="stElementContainer"]:nth-child(2){animation-delay:.08s}
.block-container [data-testid="stElementContainer"]:nth-child(3){animation-delay:.16s}
.block-container [data-testid="stElementContainer"]:nth-child(4){animation-delay:.24s}
.block-container [data-testid="stElementContainer"]:nth-child(5){animation-delay:.32s}
.block-container [data-testid="stElementContainer"]:nth-child(6){animation-delay:.40s}
.block-container [data-testid="stElementContainer"]:nth-child(7){animation-delay:.48s}
.block-container [data-testid="stElementContainer"]:nth-child(8){animation-delay:.56s}
.block-container [data-testid="stElementContainer"]:nth-child(n+9){animation-delay:.64s}
[data-testid="stColumn"] [data-testid="stElementContainer"]:nth-child(2){animation-delay:.2s}

/* cards: pale sky surface so black text stays readable */
.card {background:#E0F2FE; border-radius:14px; padding:18px 20px; margin-bottom:12px;
       border-left:5px solid #38BDF8; transition:transform .2s ease;}
.card:hover {transform:scale(1.02);}
.card, .card * {color:#000 !important;}
.card h4 {margin:0 0 6px 0; font-size:1rem; font-weight:700; color:#000 !important;}
.card p {margin:0; font-size:.92rem; line-height:1.5;}
.card.warn {border-left-color:#F97316;} .card.good {border-left-color:#22C55E;}
.lab {font-size:.8rem; font-weight:600; opacity:.7;}
.num {font-size:2.2rem; font-weight:800; line-height:1.1;}
.cw::after {counter-reset:w var(--w); content:counter(w); animation:cw 1.4s ease-out both;}
.cf::after {counter-reset:f var(--f); content:"." counter(f, decimal-leading-zero); animation:cf 1.4s ease-out both;}
@keyframes cw {from{--w:0} to{--w:var(--tw)}}
@keyframes cf {from{--f:0} to{--f:var(--tf)}}
.bar {background:#fff; border-radius:99px; height:9px; margin-top:10px; overflow:hidden;}
.bar i {display:block; height:100%; background:#38BDF8; border-radius:99px;
        animation:fill 1.4s ease-out both;}
.bar.o i {background:#F97316;} .bar.g i {background:#22C55E;}
@keyframes fill {from{width:0} to{width:var(--p)}}
.sub {color:#94A3B8 !important; font-size:1.02rem; margin:-6px 0 18px 0;}

/* plotly charts on card surface */
[data-testid="stPlotlyChart"] {background:#E0F2FE; border-radius:14px; padding:6px;}

/* buttons + sidebar nav */
.stButton>button {background:#38BDF8; color:#000; border:0; border-radius:10px; font-weight:700;
                  transition:transform .2s ease, filter .2s ease;}
.stButton>button:hover {transform:scale(1.02); filter:brightness(1.08); color:#000;}
.stButton>button p {color:#000 !important;}
section[data-testid="stSidebar"] [role="radiogroup"] {gap:4px;}
section[data-testid="stSidebar"] [role="radiogroup"] label {padding:9px 12px; border-radius:10px;
        transition:transform .2s ease, background .2s ease; cursor:pointer;}
section[data-testid="stSidebar"] [role="radiogroup"] label:hover {transform:scale(1.02); background:#1E293B;}
section[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child {display:none;}
section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {background:#38BDF8;}
section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) * {color:#000 !important; font-weight:700;}
.brand {color:#38BDF8; font-weight:800; font-size:1.05rem; line-height:1.25; margin-bottom:4px;}
@media (prefers-reduced-motion: reduce) {* {animation:none !important; transition:none !important;}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


#html helpers
def card(title, body, kind=""):
    st.markdown(f'<div class="card {kind}"><h4>{title}</h4><p>{body}</p></div>', unsafe_allow_html=True)


def kpi(label, value, suffix="", dec=0, bar=None, kind=""):
    """Animated count-up number (pure CSS) with optional progress bar (0-100)."""
    whole, frac = int(value), int(round((value - int(value)) * 100)) if dec else 0
    frac_span = '<span class="cf"></span>' if dec else ""
    bar_html = f'<div class="bar {kind}"><i style="--p:{bar:.1f}%"></i></div>' if bar is not None else ""
    st.markdown(
        f'<div class="card {"warn" if kind=="o" else "good" if kind=="g" else ""}">'
        f'<div class="lab">{label}</div>'
        f'<div class="num" style="--tw:{whole};--tf:{frac}"><span class="cw"></span>{frac_span}{suffix}</div>'
        f'{bar_html}</div>', unsafe_allow_html=True)


def header(title, sub):
    st.markdown(f"# {title}")
    st.markdown(f'<p class="sub">{sub}</p>', unsafe_allow_html=True)


def style(fig, h=340, legend=True):
    fig.update_layout(height=h, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter", color="#000"), colorway=[SKY, ORANGE, GREEN],
                      margin=dict(l=10, r=10, t=40, b=10), showlegend=legend,
                      legend=dict(orientation="h", y=-0.15))
    fig.update_xaxes(gridcolor="rgba(0,0,0,.08)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(0,0,0,.08)", zeroline=False)
    return fig


#data
@st.cache_data
def load_data(path_or_file):
    return pd.read_csv(path_or_file)


here = Path(__file__).parent
path = next((p for p in [here / "churn.csv", here / "data" / "churn.csv"] if p.exists()), None)
if path is None:
    up = st.sidebar.file_uploader("Upload churn.csv", type="csv")
    if up is None:
        st.info("Place churn.csv next to app.py (or in a data/ folder), or upload it in the sidebar.")
        st.stop()
    raw = load_data(up)
else:
    raw = load_data(path)

#cleaning + features
DROP = ["RowNumber", "CustomerId", "Surname"]


def clean(d):
    d = d.drop(columns=[c for c in DROP if c in d.columns]).drop_duplicates().dropna()
    d["AgeBand"] = pd.cut(d["Age"], [17, 30, 40, 50, 60, 100], labels=["18-30", "31-40", "41-50", "51-60", "60+"])
    return d


def features(d):
    """Model features (also used for single-customer scoring)."""
    x = d[["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard",
           "IsActiveMember", "EstimatedSalary"]].copy().astype(float)
    x["BalanceSalaryRatio"] = d["Balance"] / (d["EstimatedSalary"] + 1)
    x["TenureByAge"] = d["Tenure"] / d["Age"]
    x["HasBalance"] = (d["Balance"] > 0).astype(int)
    x["ProductsPerTenure"] = d["NumOfProducts"] / (d["Tenure"] + 1)
    for g in ["France", "Germany", "Spain"]:
        x[f"Geo_{g}"] = (d["Geography"] == g).astype(int)
    x["Male"] = (d["Gender"] == "Male").astype(int)
    return x


df = clean(raw)
X, y = features(df), df["Exited"]
ENGINEERED = ["BalanceSalaryRatio", "TenureByAge", "HasBalance", "ProductsPerTenure"]


@st.cache_resource(show_spinner="Training models...")
def train(_X, _y):
    Xtr, Xte, ytr, yte = train_test_split(_X, _y, test_size=0.2, stratify=_y, random_state=42)
    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(25)),
        "Random Forest": RandomForestClassifier(300, min_samples_leaf=5, class_weight="balanced", n_jobs=-1, random_state=42),
        "Gradient Boosting": HistGradientBoostingClassifier(learning_rate=0.05, max_iter=250, max_depth=4,
                                                             class_weight="balanced", random_state=42),
    }
    rows, probs = [], {}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        p = m.predict_proba(Xte)[:, 1]
        pred = (p >= 0.5).astype(int)
        probs[name] = p
        rows.append(dict(Model=name, Accuracy=accuracy_score(yte, pred), Precision=precision_score(yte, pred),
                         Recall=recall_score(yte, pred), F1=f1_score(yte, pred), AUC=roc_auc_score(yte, p)))
    res = pd.DataFrame(rows).sort_values("AUC", ascending=False).reset_index(drop=True)
    best_name = res.loc[0, "Model"]
    best = models[best_name]
    imp = permutation_importance(best, Xte, yte, scoring="roc_auc", n_repeats=5, random_state=42, n_jobs=-1)
    imp = pd.Series(imp.importances_mean, index=_X.columns).sort_values()
    oof = cross_val_predict(best, _X, _y, cv=5, method="predict_proba")[:, 1]
    return dict(res=res, best=best, best_name=best_name, probs=probs, yte=yte.values, imp=imp, oof=oof)


M = train(X, y)
best, res = M["best"], M["res"]
churn_rate = y.mean() * 100

#sidebar navigation
SLIDES = ["1  Executive summary", "2  Data & quality", "3  Exploratory analysis", "4  Customer segments",
          "5  Feature engineering", "6  Model comparison", "7  Evaluation & profit", "8  Risk scoring",
          "9  Decisions & actions"]
if "nav" not in st.session_state:
    st.session_state.nav = SLIDES[0]


def go_slide(step):
    st.session_state.nav = SLIDES[(SLIDES.index(st.session_state.nav) + step) % len(SLIDES)]


with st.sidebar:
    st.markdown(f'<div class="brand">🏦 {TITLE}</div>', unsafe_allow_html=True)
    st.radio("Slides", SLIDES, key="nav", label_visibility="collapsed")
    c1, c2 = st.columns(2)
    c1.button("◀ Prev", on_click=go_slide, args=(-1,), width="stretch")
    c2.button("Next ▶", on_click=go_slide, args=(1,), width="stretch")
    st.caption(f"{len(df):,} customers · best model: {M['best_name']}")

slide = SLIDES.index(st.session_state.nav)

#precomputed segment stats used across slides
df["Risk"] = M["oof"]
df["Tier"] = pd.cut(df["Risk"], [-1, .3, .6, 2], labels=["Low", "Medium", "High"])
bal_lost = df.loc[df.Exited == 1, "Balance"].sum()
seg = lambda col: df.groupby(col, observed=True)["Exited"].mean().mul(100)

#EXECUTIVE SUMMARY
if slide == 0:
    header(TITLE, "Who is leaving, why, and what the bank should do about it.")
    c = st.columns(4)
    with c[0]: kpi("Customers analysed", len(df))
    with c[1]: kpi("Churn rate", churn_rate, "%", 2, bar=churn_rate, kind="o")
    with c[2]: kpi("Balance walked out (M)", bal_lost / 1e6, "M", 2, kind="o")
    with c[3]: kpi("Best model AUC", res.loc[0, "AUC"] * 100, "%", 2, bar=res.loc[0, "AUC"] * 100, kind="g")
    geo, prod, act = seg("Geography"), seg("NumOfProducts"), seg("IsActiveMember")
    c = st.columns(3)
    with c[0]: card("Geography matters", f"{geo.idxmax()} churns at {geo.max():.1f}%, versus {geo.min():.1f}% in {geo.idxmin()}.", "warn")
    with c[1]: card("Product count is a red flag", f"Customers with {prod.idxmax()} products churn at {prod.max():.1f}%. Two products is the safest at {prod.min():.1f}%.", "warn")
    with c[2]: card("Engagement protects", f"Inactive members churn at {act[0]:.1f}% versus {act[1]:.1f}% for active members.", "good")
    card("What this deck does",
         "Cleans the data, explores it, builds and compares four models, turns the best one into a profit-based "
         "targeting rule, then ranks every customer by risk so retention teams know whom to call first.")

#DATA & QUALITY
elif slide == 1:
    header("Data & quality", "Checks before any modelling.")
    c = st.columns(4)
    with c[0]: kpi("Rows", len(raw))
    with c[1]: kpi("Columns", raw.shape[1])
    with c[2]: kpi("Missing values", int(raw.isna().sum().sum()), kind="g")
    with c[3]: kpi("Duplicate rows", int(raw.duplicated().sum()), kind="g")
    l, r = st.columns([1, 1])
    with l:
        card("Cleaning steps", "Dropped identifiers (RowNumber, CustomerId, Surname). Removed duplicates and nulls. "
             "Grouped Age into bands for reporting. Target is Exited (1 = left the bank).")
        st.dataframe(raw.dtypes.astype(str).rename("dtype").to_frame(), width="stretch", height=300)
    with r:
        st.dataframe(df.drop(columns=["Risk", "Tier"]).describe().T.round(2), width="stretch", height=420)

#EDA
elif slide == 2:
    header("Exploratory analysis", "Distribution of the target and the numeric drivers.")
    l, r = st.columns(2)
    with l:
        d = df["Exited"].map({0: "Stayed", 1: "Churned"}).value_counts()
        f = go.Figure(go.Pie(labels=d.index, values=d.values, hole=.6, marker_colors=[GREEN, ORANGE], sort=False))
        st.plotly_chart(style(f.update_layout(title="Churn split")), width="stretch")
    with r:
        f = px.histogram(df.assign(Status=df.Exited.map({0: "Stayed", 1: "Churned"})), x="Age", color="Status",
                         barmode="overlay", opacity=.75, nbins=40, color_discrete_map={"Stayed": GREEN, "Churned": ORANGE})
        st.plotly_chart(style(f.update_layout(title="Age distribution")), width="stretch")
    l, r = st.columns(2)
    with l:
        f = px.box(df.assign(Status=df.Exited.map({0: "Stayed", 1: "Churned"})), x="Status", y="Balance", color="Status",
                   color_discrete_map={"Stayed": GREEN, "Churned": ORANGE})
        st.plotly_chart(style(f.update_layout(title="Balance by outcome"), legend=False), width="stretch")
    with r:
        num = df[["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "IsActiveMember", "EstimatedSalary", "Exited"]]
        f = px.imshow(num.corr().round(2), text_auto=True, zmin=-1, zmax=1,
                      color_continuous_scale=[[0, GREEN], [.5, "#E0F2FE"], [1, ORANGE]])
        st.plotly_chart(style(f.update_layout(title="Correlation matrix", coloraxis_showscale=False)), width="stretch")
    card("Reading the charts", "Churners skew older and hold higher balances. Salary and credit score barely separate the two groups.")

#SEGMENTS
elif slide == 3:
    header("Customer segments", f"Churn rate (%) by segment. Orange bars are above the {churn_rate:.1f}% average.")
    cols = ["Geography", "Gender", "NumOfProducts", "IsActiveMember", "AgeBand", "HasCrCard"]
    for row in range(0, 6, 3):
        cc = st.columns(3)
        for col, name in zip(cc, cols[row:row + 3]):
            s = seg(name)
            f = go.Figure(go.Bar(x=s.index.astype(str), y=s.values, text=[f"{v:.1f}%" for v in s.values],
                                 textposition="outside", marker_color=[ORANGE if v > churn_rate else GREEN for v in s.values]))
            f.add_hline(y=churn_rate, line_dash="dot", line_color=SKY)
            f.update_yaxes(range=[0, max(s.max() * 1.25, 10)])
            col.plotly_chart(style(f.update_layout(title=name), h=300, legend=False), width="stretch")
    card("So what", "Risk concentrates in Germany, in customers aged 41-60, in those holding 3-4 products, and in inactive members.", "warn")

#FEATURE ENGINEERING
elif slide == 4:
    header("Feature engineering", "Four ratios added to the raw fields, plus one-hot country and gender.")
    l, r = st.columns([1, 1.2])
    with l:
        card("BalanceSalaryRatio", "Balance divided by salary: how much of their income sits with us.")
        card("TenureByAge", "Share of life spent as a customer: loyalty independent of age.")
        card("HasBalance", "Flags customers with a zero balance.")
        card("ProductsPerTenure", "How fast products were added relative to time with the bank.")
        card("Pipeline", "Stratified 80/20 split, class weights for the 20% minority, scaling for distance-based models.")
    with r:
        corr = X.assign(Exited=y).corr()["Exited"].drop("Exited").sort_values()
        f = go.Figure(go.Bar(x=corr.values, y=corr.index, orientation="h",
                             marker_color=[ORANGE if v > 0 else GREEN for v in corr.values]))
        f.update_layout(title="Correlation of each feature with churn")
        st.plotly_chart(style(f, h=520, legend=False), width="stretch")

#MODEL COMPARISON
elif slide == 5:
    header("Model comparison", "Four algorithms, same split, same features. Ranked by ROC-AUC.")
    c = st.columns(4)
    top = res.iloc[0]
    with c[0]: kpi(f"{top.Model}: AUC", top.AUC * 100, "%", 2, bar=top.AUC * 100, kind="g")
    with c[1]: kpi("Recall", top.Recall * 100, "%", 2, bar=top.Recall * 100)
    with c[2]: kpi("Precision", top.Precision * 100, "%", 2, bar=top.Precision * 100)
    with c[3]: kpi("F1", top.F1 * 100, "%", 2, bar=top.F1 * 100)
    l, r = st.columns(2)
    with l:
        long = res.melt("Model", var_name="Metric", value_name="Score")
        f = px.bar(long, x="Metric", y="Score", color="Model", barmode="group",
                   color_discrete_sequence=[SKY, ORANGE, GREEN, "#0F172A"])
        st.plotly_chart(style(f.update_layout(title="Metrics by model")), width="stretch")
    with r:
        f = go.Figure()
        for (name, p), colr in zip(M["probs"].items(), [SKY, ORANGE, GREEN, "#0F172A"]):
            fpr, tpr, _ = roc_curve(M["yte"], p)
            f.add_scatter(x=fpr, y=tpr, name=name, line=dict(color=colr, width=3))
        f.add_scatter(x=[0, 1], y=[0, 1], line=dict(dash="dot", color="grey"), showlegend=False)
        f.update_layout(title="ROC curves", xaxis_title="False positive rate", yaxis_title="True positive rate")
        st.plotly_chart(style(f), width="stretch")
    st.dataframe(res.style.format({c: "{:.3f}" for c in res.columns[1:]}), width="stretch", hide_index=True)

#EVALUATION & PROFIT
elif slide == 6:
    header("Evaluation & profit", f"{M['best_name']}: what drives churn and where to set the contact threshold.")
    l, r = st.columns(2)
    p, yt = M["probs"][M["best_name"]], M["yte"]
    with l:
        cm = confusion_matrix(yt, (p >= .5).astype(int))
        f = px.imshow(cm, text_auto=True, x=["Stay", "Churn"], y=["Stayed", "Churned"],
                      color_continuous_scale=[[0, "#E0F2FE"], [1, SKY]], labels=dict(x="Predicted", y="Actual"))
        st.plotly_chart(style(f.update_layout(title="Confusion matrix (threshold 0.50)", coloraxis_showscale=False)), width="stretch")
    with r:
        imp = M["imp"].tail(10)
        f = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color=SKY))
        f.update_layout(title="Top drivers (permutation importance, AUC drop)")
        st.plotly_chart(style(f, legend=False), width="stretch")

    st.markdown("### Retention budget simulator")
    a, b, c3 = st.columns(3)
    cost = a.slider("Cost per retention offer", 10, 500, 60)
    value = b.slider("Value of a saved customer", 100, 5000, 800, step=50)
    succ = c3.slider("Offer success rate (%)", 5, 80, 30) / 100
    scale = len(df) / len(yt)
    ts = np.linspace(.05, .95, 91)
    prof = []
    for t in ts:
        pred = p >= t
        tp, fp = (pred & (yt == 1)).sum(), (pred & (yt == 0)).sum()
        prof.append((tp * succ * value - (tp + fp) * cost) * scale)
    prof = np.array(prof)
    k = prof.argmax()
    f = go.Figure(go.Scatter(x=ts, y=prof, line=dict(color=GREEN, width=3), fill="tozeroy", fillcolor="rgba(34,197,94,.2)"))
    f.add_scatter(x=[ts[k]], y=[prof[k]], mode="markers", marker=dict(size=14, color=ORANGE), name="Best")
    f.update_layout(title="Expected campaign profit by contact threshold", xaxis_title="Probability threshold", yaxis_title="Profit")
    st.plotly_chart(style(f, legend=False), width="stretch")
    n_contact = int((p >= ts[k]).sum() * scale)
    card("Recommendation", f"Contact customers scoring above <b>{ts[k]:.2f}</b> (about {n_contact:,} people). "
         f"Expected profit under these assumptions: <b>{prof[k]:,.0f}</b>.", "good" if prof[k] > 0 else "warn")

#RISK SCORING
elif slide == 7:
    header("Risk scoring", "Describe a customer and get a live churn risk and next best action.")
    l, r = st.columns([1.1, 1])
    with l:
        a, b = st.columns(2)
        geo = a.selectbox("Country", ["France", "Germany", "Spain"])
        gen = b.radio("Gender", ["Female", "Male"], horizontal=True)
        age = a.slider("Age", 18, 90, 42)
        ten = b.slider("Tenure (years)", 0, 10, 4)
        score = a.slider("Credit score", 300, 850, 650)
        prods = b.slider("Products", 1, 4, 1)
        bal = a.number_input("Balance", 0.0, 300000.0, 90000.0, step=5000.0)
        sal = b.number_input("Estimated salary", 0.0, 250000.0, 100000.0, step=5000.0)
        card_ = a.checkbox("Has credit card", True)
        active = b.checkbox("Active member", True)
    row = pd.DataFrame([dict(CreditScore=score, Geography=geo, Gender=gen, Age=age, Tenure=ten, Balance=bal,
                             NumOfProducts=prods, HasCrCard=int(card_), IsActiveMember=int(active), EstimatedSalary=sal)])
    risk = float(best.predict_proba(features(row))[0, 1])
    with r:
        f = go.Figure(go.Indicator(mode="gauge+number", value=risk * 100, number=dict(suffix="%"),
                                   gauge=dict(axis=dict(range=[0, 100]), bar=dict(color="#0F172A"),
                                              steps=[dict(range=[0, 30], color=GREEN), dict(range=[30, 60], color="#FDBA74"),
                                                     dict(range=[60, 100], color=ORANGE)])))
        st.plotly_chart(style(f.update_layout(title="Churn probability"), h=300, legend=False), width="stretch")
        if risk >= .6:
            card("High risk", "Assign a relationship manager this week. Offer a tailored package and review product fit.", "warn")
        elif risk >= .3:
            card("Medium risk", "Send a personalised engagement offer and invite them to use an unused product.")
        else:
            card("Low risk", "Keep servicing as normal. Good candidate for a cross-sell conversation.", "good")

#DECISIONS
else:
    header("Decisions & actions", "Every customer scored out-of-fold, then grouped into action tiers.")
    t = df.groupby("Tier", observed=True).agg(Customers=("Exited", "size"), ActualChurn=("Exited", "mean"),
                                              Balance=("Balance", "sum")).reset_index()
    t["ActualChurn"] *= 100
    colors = {"Low": GREEN, "Medium": "#FDBA74", "High": ORANGE}
    c = st.columns(3)
    for col, (_, rw) in zip(c, t.iterrows()):
        with col:
            kpi(f"{rw.Tier} risk customers", rw.Customers, bar=rw.ActualChurn,
                kind="g" if rw.Tier == "Low" else "o" if rw.Tier == "High" else "")
            st.caption(f"{rw.ActualChurn:.1f}% actually churned · balances {rw.Balance/1e6:.1f}M")
    l, r = st.columns(2)
    with l:
        f = go.Figure(go.Bar(x=t.Tier.astype(str), y=t.Balance / 1e6, marker_color=[colors[x] for x in t.Tier.astype(str)],
                             text=[f"{v:.1f}M" for v in t.Balance / 1e6], textposition="outside"))
        f.update_layout(title="Balances held by risk tier (M)")
        st.plotly_chart(style(f, legend=False), width="stretch")
    with r:
        card("1. Win back Germany", "Highest-churn market. Launch a local retention programme and investigate pricing and service gaps.", "warn")
        card("2. Re-engage inactive members", "Trigger campaigns on dormant accounts before they close.", "warn")
        card("3. Fix the 3-4 product bundles", "These customers leave most often. Review fees and product fit.", "warn")
        card("4. Target ages 41-60", "Build retirement and savings propositions for the highest-risk age bands.")
    st.markdown("### Top 25 customers to contact")
    top = raw.loc[df.index].assign(ChurnRisk=(df["Risk"] * 100).round(1))
    top = top[top.Exited == 0].sort_values("ChurnRisk", ascending=False).head(25)
    st.dataframe(top[["CustomerId", "Surname", "Geography", "Age", "NumOfProducts", "IsActiveMember", "Balance", "ChurnRisk"]],
                 width="stretch", hide_index=True)
