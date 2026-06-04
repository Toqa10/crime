import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Crime Predictor · LA",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── GLOBAL CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Base ── */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&family=JetBrains+Mono:wght@400;600&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .stApp { background: #080c14; }

  /* ── Hide default streamlit elements ── */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding: 0 2rem 3rem 2rem; max-width: 1300px; }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #0f1923 100%);
    border-right: 1px solid #1e293b;
  }
  [data-testid="stSidebar"] .stSelectbox label,
  [data-testid="stSidebar"] .stSlider label,
  [data-testid="stSidebar"] .stNumberInput label { color: #94a3b8; font-size: 0.8rem; letter-spacing: 0.05em; text-transform: uppercase; }

  /* ── Metric cards ── */
  [data-testid="metric-container"] {
    background: #0d1117;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 1rem;
  }
  [data-testid="metric-container"] label { color: #64748b !important; font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 0.08em; }
  [data-testid="metric-container"] [data-testid="stMetricValue"] { color: #e2e8f0 !important; font-weight: 700 !important; }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] { background: #0d1117; border-bottom: 1px solid #1e293b; gap: 0; }
  .stTabs [data-baseweb="tab"] { background: transparent; color: #64748b; border: none; padding: 0.75rem 1.5rem; font-size: 0.85rem; font-weight: 600; letter-spacing: 0.05em; }
  .stTabs [aria-selected="true"] { color: #e94560 !important; border-bottom: 2px solid #e94560 !important; background: transparent !important; }

  /* ── Buttons ── */
  .stButton > button {
    background: linear-gradient(135deg, #e94560, #c73652);
    color: white; border: none; border-radius: 8px;
    padding: 0.65rem 2rem; font-weight: 700;
    font-size: 0.95rem; letter-spacing: 0.05em;
    transition: all 0.2s ease; width: 100%;
    box-shadow: 0 4px 20px rgba(233,69,96,0.3);
  }
  .stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(233,69,96,0.5);
  }

  /* ── Selectbox & inputs ── */
  .stSelectbox > div > div,
  .stNumberInput > div > div > input {
    background: #0d1117 !important;
    border: 1px solid #1e293b !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
  }

  /* ── Dataframe ── */
  .stDataFrame { border: 1px solid #1e293b; border-radius: 12px; overflow: hidden; }

  /* ── Custom card ── */
  .crime-card {
    background: #0d1117;
    border: 1px solid #1e293b;
    border-radius: 14px;
    padding: 1.5rem;
    margin-bottom: 1rem;
  }
  .crime-card-red { border-left: 3px solid #e94560; }
  .crime-card-blue { border-left: 3px solid #3b82f6; }
  .crime-card-green { border-left: 3px solid #22c55e; }
  .crime-card-purple { border-left: 3px solid #a855f7; }

  /* ── Prediction result ── */
  .pred-violent {
    background: linear-gradient(135deg, #1a0a0e, #2d0f18);
    border: 2px solid #e94560;
    border-radius: 16px; padding: 2rem; text-align: center;
    box-shadow: 0 0 40px rgba(233,69,96,0.2);
  }
  .pred-nonviolent {
    background: linear-gradient(135deg, #0a1a10, #0f2d1a);
    border: 2px solid #22c55e;
    border-radius: 16px; padding: 2rem; text-align: center;
    box-shadow: 0 0 40px rgba(34,197,94,0.2);
  }
  .pred-title { font-size: 2.5rem; font-weight: 900; margin: 0; letter-spacing: 2px; }
  .pred-sub   { color: #94a3b8; font-size: 0.9rem; margin-top: 0.5rem; }

  /* ── Section headers ── */
  .section-header {
    font-size: 1.1rem; font-weight: 700;
    color: #94a3b8; letter-spacing: 0.1em;
    text-transform: uppercase;
    border-bottom: 1px solid #1e293b;
    padding-bottom: 0.5rem; margin-bottom: 1.5rem;
  }
</style>
""", unsafe_allow_html=True)

# ─── DATA & MODEL LOADERS ─────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("Sample_200k.csv")
    df["DATE OCC"] = pd.to_datetime(df["DATE OCC"], errors="coerce")
    df["Hour OCC"] = df["TIME OCC"].apply(lambda x: int(str(int(x)).zfill(4)[:2]))
    df["Month OCC"] = df["DATE OCC"].dt.month
    df["Year OCC"]  = df["DATE OCC"].dt.year
    df["day_of_week OCC"] = df["DATE OCC"].dt.day_of_week
    df["Weapon Desc"]  = df["Weapon Desc"].fillna("No Weapon")
    df["Vict Descent"] = df["Vict Descent"].fillna("Unknown")
    mapping = {1: "Violent", 2: "Non-Violent"}
    df["Crime_Type"] = df["Part 1-2"].map(mapping)
    return df

@st.cache_resource
def load_models():
    files = {
        "LogisticRegression":    "Files/LogisticRegression.pkl",
        "RandomForest":          "Files/RandomForestClassifier.pkl",
        "XGBoost":               "Files/XGBoost.pkl",
        "LightGBM":              "Files/LightGBM.pkl",
        "CatBoost":              "Files/CatBoost.pkl",
        "DecisionTree":          "Files/DecisionTreeClassifier.pkl",
        "ExtraTrees":            "Files/ExtraTreesClassifier.pkl",
        "KNN":                   "Files/KNeighborsClassifier.pkl",
        "GaussianNB":            "Files/GaussianNB.pkl",
        "Bagging":               "Files/Bagging.pkl",
        "AdaBoost":              "Files/AdaBoost.pkl",
    }
    loaded = {}
    for name, path in files.items():
        try:
            with open(path, "rb") as f:
                loaded[name] = pickle.load(f)
        except Exception:
            pass
    return loaded

@st.cache_resource
def load_preprocessors():
    preps = {}
    paths = {
        "le_y":         "Files/le_y.pkl",
        "yeo":          "Files/yeo-johnson.pkl",
        "std":          "Files/StandardScaler.pkl",
        "std2":         "Files/StandardScaler2.pkl",
        "robust":       "Files/RobustScaler.pkl",
        "le_crm":       "Files/Crm Cd Desc_label_encoder.pkl",
        "le_descent":   "Files/Vict Descent_label_encoder.pkl",
        "le_weapon":    "Files/Weapon Desc_label_encoder.pkl",
        "le_status":    "Files/Status Desc_label_encoder.pkl",
        "imputer":      "Files/Iterative_imputer_Age.pkl",
    }
    for name, path in paths.items():
        try:
            with open(path, "rb") as f:
                preps[name] = pickle.load(f)
        except Exception:
            pass
    return preps

# ─── PLOTLY THEME ─────────────────────────────────────────────────────────────
DARK_BG   = "#080c14"
CARD_BG   = "#0d1117"
BORDER    = "#1e293b"
RED       = "#e94560"
BLUE      = "#3b82f6"
GREEN     = "#22c55e"
PURPLE    = "#a855f7"
TEXT      = "#e2e8f0"
MUTED     = "#64748b"

def chart_layout(fig, title="", h=380):
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=TEXT), x=0),
        plot_bgcolor=CARD_BG, paper_bgcolor=CARD_BG,
        font=dict(color=TEXT, family="Inter"),
        margin=dict(l=20, r=20, t=50 if title else 20, b=20),
        height=h,
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
        xaxis=dict(gridcolor=BORDER, linecolor=BORDER, tickfont=dict(color=MUTED)),
        yaxis=dict(gridcolor=BORDER, linecolor=BORDER, tickfont=dict(color=MUTED)),
    )
    return fig

# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#080c14 0%,#0d1117 50%,#0f1923 100%);
            padding:2.5rem 2rem 2rem; border-bottom:1px solid #1e293b; margin-bottom:2rem;">
  <div style="display:flex; align-items:center; gap:1rem;">
    <div style="background:linear-gradient(135deg,#e94560,#c73652);
                width:52px;height:52px;border-radius:14px;
                display:flex;align-items:center;justify-content:center;
                font-size:1.8rem;box-shadow:0 0 20px rgba(233,69,96,0.4);">🔍</div>
    <div>
      <h1 style="color:#e2e8f0;margin:0;font-size:1.9rem;font-weight:900;letter-spacing:1px;">
        CRIME PREDICTOR
      </h1>
      <p style="color:#64748b;margin:0;font-size:0.85rem;letter-spacing:3px;text-transform:uppercase;">
        Los Angeles · LAPD Dataset · 2020–Present
      </p>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── LOAD DATA ────────────────────────────────────────────────────────────────
df = load_data()
models = load_models()
preps  = load_preprocessors()

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:1rem 0 1.5rem;">
      <div style="font-size:2rem;">🗺️</div>
      <div style="color:#e2e8f0;font-weight:700;font-size:1.1rem;margin-top:0.3rem;">Navigation</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("", ["📊 Dashboard", "🤖 Predict Crime", "📈 Model Results"], label_visibility="collapsed")

    st.markdown("---")
    st.markdown('<div style="color:#64748b;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.75rem;">Quick Stats</div>', unsafe_allow_html=True)

    total = len(df)
    violent = (df["Part 1-2"] == 1).sum()
    areas   = df["AREA NAME"].nunique()
    years   = df["Year OCC"].nunique()

    for label, val in [("Total Records", f"{total:,}"), ("Violent Crimes", f"{violent:,}"),
                        ("LAPD Areas", str(areas)),  ("Years Covered", str(years))]:
        st.markdown(f"""
        <div style="background:#0d1117;border:1px solid #1e293b;border-radius:8px;
                    padding:0.6rem 0.9rem;margin-bottom:0.5rem;display:flex;
                    justify-content:space-between;align-items:center;">
          <span style="color:#64748b;font-size:0.75rem;">{label}</span>
          <span style="color:#e2e8f0;font-weight:700;font-size:0.9rem;">{val}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f'<div style="color:#334155;font-size:0.7rem;text-align:center;">Models loaded: {len(models)}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 – DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Dashboard":

    # KPI row
    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        ("Total Crimes",    f"{total:,}",                           "🔴"),
        ("Violent",         f"{violent:,}",                         "⚠️"),
        ("Non-Violent",     f"{(df['Part 1-2']==2).sum():,}",       "🟢"),
        ("LAPD Areas",      str(areas),                             "🗺️"),
        ("Crime Types",     str(df["Crm Cd Desc"].nunique()),        "📋"),
    ]
    for col, (label, val, icon) in zip([c1,c2,c3,c4,c5], kpis):
        col.metric(f"{icon} {label}", val)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 1: Violent vs Non-Violent + Monthly trend ──────────────────────
    col1, col2 = st.columns([1, 2])

    with col1:
        counts = df["Crime_Type"].value_counts()
        fig = go.Figure(go.Pie(
            labels=counts.index, values=counts.values,
            hole=0.6,
            marker=dict(colors=[RED, BLUE], line=dict(color=DARK_BG, width=3)),
            textfont=dict(color=TEXT, size=13),
        ))
        fig.add_annotation(text=f"<b>{total:,}</b><br><span style='font-size:11px'>Total</span>",
                           x=0.5, y=0.5, showarrow=False,
                           font=dict(size=16, color=TEXT))
        chart_layout(fig, "🎯 Crime Type Split")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        monthly = df.groupby(df["DATE OCC"].dt.to_period("M").dt.to_timestamp()).size().reset_index(name="count")
        fig = px.line(monthly, x="DATE OCC", y="count", markers=True,
                      color_discrete_sequence=[RED])
        fig.update_traces(line_width=2.5, marker=dict(size=5, color=RED))
        fig.add_hrect(y0=monthly["count"].mean()-monthly["count"].std(),
                      y1=monthly["count"].mean()+monthly["count"].std(),
                      fillcolor=BLUE, opacity=0.08, line_width=0)
        chart_layout(fig, "📈 Monthly Crime Trend")
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 2: Heatmap + Area bar ──────────────────────────────────────────
    col1, col2 = st.columns([3, 2])

    with col1:
        pivot = df.pivot_table(values="Crm Cd 1", index="day_of_week OCC",
                               columns="Hour OCC", aggfunc="count", fill_value=0)
        pivot.index = pivot.index.map({0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"})
        pivot_long = pivot.reset_index().melt(id_vars="day_of_week OCC", var_name="Hour", value_name="Count")
        fig = px.density_heatmap(pivot_long, x="Hour", y="day_of_week OCC", z="Count",
                                  color_continuous_scale="Reds",
                                  labels={"day_of_week OCC": "Day", "Count": "Crimes"})
        chart_layout(fig, "🔥 Crime Density: Day × Hour")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        area = df["AREA NAME"].value_counts().head(10).sort_values()
        fig = go.Figure(go.Bar(
            x=area.values, y=area.index, orientation="h",
            marker=dict(
                color=area.values,
                colorscale=[[0, BLUE], [0.5, PURPLE], [1, RED]],
                showscale=False
            ),
            text=area.values, textposition="outside",
            textfont=dict(color=MUTED, size=10),
        ))
        chart_layout(fig, "🗺️ Top 10 Areas by Crime Count")
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 3: Top crime types + Victim age ──────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        top = df["Crm Cd Desc"].value_counts().head(12).sort_values()
        colors = [RED if i >= 9 else (PURPLE if i >= 6 else BLUE) for i in range(12)]
        fig = go.Figure(go.Bar(x=top.values, y=top.index, orientation="h",
                               marker_color=colors))
        chart_layout(fig, "📋 Top 12 Crime Types", h=420)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        df_age = df[(df["Vict Age"] > 10) & (df["Vict Age"] < 90)]
        fig = px.histogram(df_age, x="Vict Age", nbins=35,
                           color="Crime_Type",
                           color_discrete_map={"Violent": RED, "Non-Violent": BLUE},
                           barmode="overlay", opacity=0.75)
        chart_layout(fig, "👤 Victim Age by Crime Type", h=420)
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 4: Weapon + Descent ──────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        wp = df[df["Weapon Desc"] != "No Weapon"]["Weapon Desc"].value_counts().head(8)
        fig = px.bar(x=wp.values, y=wp.index, orientation="h",
                     color=wp.values, color_continuous_scale="Reds")
        fig.update_coloraxes(showscale=False)
        chart_layout(fig, "🔫 Top Weapons Used")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        descent_map = {"H":"Hispanic","W":"White","B":"Black","A":"Other Asian",
                       "C":"Chinese","X":"Unknown","O":"Other","J":"Japanese",
                       "V":"Vietnamese","K":"Korean","F":"Filipino","I":"Native American"}
        des = df["Vict Descent"].map(descent_map).fillna("Other").value_counts().head(8)
        fig = px.pie(values=des.values, names=des.index, hole=0.45,
                     color_discrete_sequence=[RED,BLUE,PURPLE,GREEN,"#f59e0b","#06b6d4","#ec4899","#84cc16"])
        chart_layout(fig, "🌍 Victim Descent Distribution")
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 – PREDICT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Predict Crime":

    st.markdown('<h2 style="color:#e2e8f0;font-weight:800;margin-bottom:0.3rem;">🤖 Crime Classification</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color:#64748b;margin-bottom:2rem;">Enter crime details to predict whether it is Violent or Non-Violent</p>', unsafe_allow_html=True)

    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown('<div class="section-header">Crime Details</div>', unsafe_allow_html=True)

        crm_desc_options = sorted(df["Crm Cd Desc"].dropna().unique().tolist())
        crm_desc = st.selectbox("Crime Description", crm_desc_options)

        weapon_options = sorted(df["Weapon Desc"].dropna().unique().tolist())
        weapon_desc = st.selectbox("Weapon Used", weapon_options)

        status_options = sorted(df["Status Desc"].dropna().unique().tolist())
        status_desc = st.selectbox("Case Status", status_options)

        st.markdown('<div class="section-header" style="margin-top:1.5rem;">Victim Info</div>', unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            vict_age = st.number_input("Victim Age", min_value=0, max_value=100, value=30)
            vict_sex = st.selectbox("Victim Sex", ["Male (M)", "Female (F)", "Unknown (X)"])
        with col_b:
            descent_map_rev = {"Hispanic":"H","White":"W","Black":"B","Asian":"A",
                               "Chinese":"C","Unknown":"X","Other":"O"}
            vict_descent_label = st.selectbox("Victim Descent",
                list(descent_map_rev.keys()))
            vict_descent = descent_map_rev[vict_descent_label]

        st.markdown('<div class="section-header" style="margin-top:1.5rem;">Location & Codes</div>', unsafe_allow_html=True)
        col_c, col_d = st.columns(2)
        with col_c:
            crm_cd1   = st.number_input("Crime Code 1", min_value=100, max_value=956, value=624)
            premis_cd = st.number_input("Premise Code",  min_value=100, max_value=956, value=501)
        with col_d:
            weapon_cd = st.number_input("Weapon Code", min_value=0, max_value=999, value=400)

        selected_model = st.selectbox("🤖 Model", list(models.keys()) if models else ["No models loaded"])

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("🔍 PREDICT", use_container_width=True)

    # ── Result panel ───────────────────────────────────────────────────────
    with col_result:
        st.markdown('<div class="section-header">Prediction Result</div>', unsafe_allow_html=True)

        if predict_btn and models:
            try:
                sex_map = {"Male (M)": 0, "Female (F)": 1, "Unknown (X)": 3}
                sex_enc = sex_map[vict_sex]

                # Encode categorical with saved label encoders
                def safe_encode(le, val, default=0):
                    if le is None: return default
                    classes = list(le.classes_)
                    return int(le.transform([val])[0]) if val in classes else default

                crm_enc     = safe_encode(preps.get("le_crm"),     crm_desc)
                weapon_enc  = safe_encode(preps.get("le_weapon"),  weapon_desc)
                status_enc  = safe_encode(preps.get("le_status"),  status_desc)
                descent_enc = safe_encode(preps.get("le_descent"), vict_descent)

                # Build feature array (same order as training)
                # FEATURES = ['Crm Cd 1','Crm Cd Desc','Weapon Used Cd','Premis Cd',
                #             'Weapon Desc','Vict Age','Vict Descent','Vict Sex','Status Desc']
                features_raw = np.array([[crm_cd1, crm_enc, weapon_cd, premis_cd,
                                          weapon_enc, vict_age, descent_enc, sex_enc, status_enc]],
                                        dtype=float)

                import pandas as pd
                FEAT_NAMES = ['Crm Cd 1','Crm Cd Desc','Weapon Used Cd','Premis Cd',
                              'Weapon Desc','Vict Age','Vict Descent','Vict Sex','Status Desc']
                X_input = pd.DataFrame(features_raw, columns=FEAT_NAMES)

                # Apply scalers
                yeo_cols    = ['Weapon Desc','Status Desc','Vict Sex']
                std_cols    = ['Crm Cd Desc','Vict Descent']
                log_cols    = ['Crm Cd 1','Premis Cd']
                robust_cols = ['Vict Age','Weapon Used Cd']

                if preps.get("yeo"):
                    X_input[yeo_cols] = preps["yeo"].transform(X_input[yeo_cols])
                if preps.get("std"):
                    X_input[std_cols] = preps["std"].transform(X_input[std_cols])
                if preps.get("std2"):
                    X_input[log_cols] = np.log1p(X_input[log_cols])
                    X_input[log_cols] = preps["std2"].transform(X_input[log_cols])
                if preps.get("robust"):
                    X_input[robust_cols] = preps["robust"].transform(X_input[robust_cols])

                model = models[selected_model]
                pred = model.predict(X_input)[0]
                label = "Violent" if pred == 1 else "Non-Violent"

                # Probability if available
                prob = None
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X_input)[0]
                    prob  = proba[pred]

                is_violent = (label == "Violent")
                card_class = "pred-violent" if is_violent else "pred-nonviolent"
                color      = RED if is_violent else GREEN
                icon       = "⚠️" if is_violent else "✅"

                st.markdown(f"""
                <div class="{card_class}">
                  <div style="font-size:3rem;margin-bottom:0.5rem;">{icon}</div>
                  <div class="pred-title" style="color:{color};">{label.upper()}</div>
                  <div class="pred-sub">Classification result using {selected_model}</div>
                  {"" if prob is None else f'<div style="margin-top:1rem;color:{color};font-size:1.5rem;font-weight:700;">{prob*100:.1f}% Confidence</div>'}
                </div>
                """, unsafe_allow_html=True)

                # Gauge chart for confidence
                if prob is not None:
                    st.markdown("<br>", unsafe_allow_html=True)
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=prob * 100,
                        number=dict(suffix="%", font=dict(size=28, color=color)),
                        gauge=dict(
                            axis=dict(range=[0, 100], tickcolor=MUTED),
                            bar=dict(color=color),
                            bgcolor=CARD_BG,
                            bordercolor=BORDER,
                            steps=[dict(range=[0,50], color="#0d1117"),
                                   dict(range=[50,75], color="#1a1a2e"),
                                   dict(range=[75,100], color="#1a0a0e" if is_violent else "#0a1a10")],
                            threshold=dict(line=dict(color=color, width=3), value=prob*100)
                        )
                    ))
                    fig.update_layout(
                        paper_bgcolor=CARD_BG, font=dict(color=TEXT),
                        height=220, margin=dict(l=30, r=30, t=30, b=10)
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Feature summary table
                st.markdown('<div class="section-header" style="margin-top:1rem;">Input Summary</div>', unsafe_allow_html=True)
                summary = pd.DataFrame({
                    "Feature": ["Crime Type", "Weapon", "Status", "Age", "Sex", "Descent"],
                    "Value":   [crm_desc, weapon_desc, status_desc,
                                str(vict_age), vict_sex.split("(")[0].strip(), vict_descent_label]
                })
                st.dataframe(summary, use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(f"❌ Prediction error: {e}")
                st.info("Make sure all model files are in the `Files/` folder.")

        elif predict_btn and not models:
            st.warning("⚠️ No models loaded. Run the notebook first to generate Files/")
        else:
            st.markdown("""
            <div style="background:#0d1117;border:1px dashed #1e293b;border-radius:14px;
                        padding:3rem;text-align:center;margin-top:1rem;">
              <div style="font-size:3rem;margin-bottom:1rem;">🤖</div>
              <div style="color:#475569;font-size:0.95rem;">
                Fill in the crime details on the left<br>and click <b style="color:#e94560;">PREDICT</b>
              </div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 – MODEL RESULTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Results":

    st.markdown('<h2 style="color:#e2e8f0;font-weight:800;margin-bottom:0.3rem;">📈 Model Performance</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color:#64748b;margin-bottom:2rem;">Evaluation metrics for all 11 trained classifiers</p>', unsafe_allow_html=True)

    # Static results table from notebook
    results_data = {
        "Model":     ["XGBoost","LightGBM","CatBoost","RandomForest","ExtraTrees",
                      "DecisionTree","Bagging","AdaBoost","KNN",
                      "LogisticRegression","GaussianNB"],
        "Accuracy":  [0.9999,0.9999,0.9999,0.9999,0.9999,0.9999,0.9999,0.9890,0.9926,0.8855,0.8697],
        "Precision": [0.9999,0.9999,0.9999,0.9999,0.9999,0.9999,0.9999,0.9807,0.9901,0.8831,0.8884],
        "Recall":    [1.0000,1.0000,0.9999,0.9999,0.9998,0.9999,0.9999,0.9920,0.9913,0.8230,0.7713],
        "F1":        [0.9999,0.9999,0.9999,0.9999,0.9999,0.9999,0.9999,0.9863,0.9907,0.8520,0.8257],
    }
    results_df = pd.DataFrame(results_data)

    # Try to load from pickle if available
    try:
        with open("Files/results_dict.pkl", "rb") as f:
            results_dict = pickle.load(f)
        rows = []
        for name, vals in results_dict.items():
            m = vals["metrics"]
            rows.append([name, m["Accuracy"], m["Precision"], m["Recall"], m["F1"]])
        results_df = pd.DataFrame(rows, columns=["Model","Accuracy","Precision","Recall","F1"])
        results_df = results_df.sort_values("Accuracy", ascending=False).reset_index(drop=True)
    except Exception:
        pass

    # ── Top 3 podium ──────────────────────────────────────────────────────
    podium_colors = [("#f59e0b","🥇"), (MUTED,"🥈"), ("#cd7c3f","🥉")]
    col1, col2, col3 = st.columns(3)
    for col, i, (color, medal) in zip([col1,col2,col3], range(3), podium_colors):
        row = results_df.iloc[i]
        col.markdown(f"""
        <div style="background:#0d1117;border:1px solid {color};border-radius:14px;
                    padding:1.5rem;text-align:center;box-shadow:0 0 20px {color}22;">
          <div style="font-size:2rem;">{medal}</div>
          <div style="color:{color};font-weight:800;font-size:1.05rem;margin:0.5rem 0;">{row['Model']}</div>
          <div style="color:#e2e8f0;font-size:1.8rem;font-weight:900;">{row['Accuracy']*100:.2f}%</div>
          <div style="color:#64748b;font-size:0.75rem;margin-top:0.3rem;">Accuracy</div>
          <div style="display:flex;justify-content:space-around;margin-top:1rem;">
            <div><div style="color:{color};font-weight:700;">{row['F1']*100:.2f}%</div><div style="color:#475569;font-size:0.7rem;">F1</div></div>
            <div><div style="color:{color};font-weight:700;">{row['Recall']*100:.2f}%</div><div style="color:#475569;font-size:0.7rem;">Recall</div></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Bar comparison ────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        for metric, color in zip(["Accuracy","F1","Precision","Recall"],
                                  [RED, BLUE, PURPLE, GREEN]):
            fig.add_trace(go.Bar(name=metric, x=results_df["Model"],
                                 y=results_df[metric], marker_color=color, opacity=0.85))
        fig.update_layout(barmode="group", xaxis_tickangle=-35)
        chart_layout(fig, "📊 All Metrics Comparison", h=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=[results_df.iloc[0]["Accuracy"],
               results_df.iloc[0]["Precision"],
               results_df.iloc[0]["Recall"],
               results_df.iloc[0]["F1"],
               results_df.iloc[0]["Accuracy"]],
            theta=["Accuracy","Precision","Recall","F1","Accuracy"],
            fill="toself", name=results_df.iloc[0]["Model"],
            line_color=RED, fillcolor=f"{RED}22"
        ))
        fig.add_trace(go.Scatterpolar(
            r=[results_df.iloc[-1]["Accuracy"],
               results_df.iloc[-1]["Precision"],
               results_df.iloc[-1]["Recall"],
               results_df.iloc[-1]["F1"],
               results_df.iloc[-1]["Accuracy"]],
            theta=["Accuracy","Precision","Recall","F1","Accuracy"],
            fill="toself", name=results_df.iloc[-1]["Model"],
            line_color=BLUE, fillcolor=f"{BLUE}22"
        ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0.7,1.0], color=MUTED, gridcolor=BORDER),
                angularaxis=dict(color=TEXT, gridcolor=BORDER),
                bgcolor=CARD_BG
            ),
            paper_bgcolor=CARD_BG, font=dict(color=TEXT),
            height=400, margin=dict(l=40,r=40,t=50,b=20),
            title=dict(text="🕸️ Best vs Worst – Radar", font=dict(color=TEXT, size=15)),
            legend=dict(bgcolor="rgba(0,0,0,0)")
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Full results table ────────────────────────────────────────────────
    st.markdown('<div class="section-header">Full Results Table</div>', unsafe_allow_html=True)

    styled = results_df.copy()
    for col in ["Accuracy","Precision","Recall","F1"]:
        styled[col] = styled[col].map(lambda x: f"{x*100:.2f}%")
    styled.insert(0, "Rank", range(1, len(styled)+1))
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # ── Accuracy horizontal bar ───────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    fig = go.Figure(go.Bar(
        x=results_df["Accuracy"],
        y=results_df["Model"],
        orientation="h",
        marker=dict(
            color=results_df["Accuracy"],
            colorscale=[[0, BLUE], [0.5, PURPLE], [1, RED]],
            showscale=False
        ),
        text=[f"{v*100:.2f}%" for v in results_df["Accuracy"]],
        textposition="outside",
        textfont=dict(color=MUTED, size=11),
    ))
    fig.update_layout(xaxis_range=[0.75, 1.03])
    chart_layout(fig, "🏆 Accuracy Ranking – All Models", h=420)
    st.plotly_chart(fig, use_container_width=True)
