import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit.components.v1 as components

# ---------------------------------
# Batman Theme (Dark Gotham Style)
# ---------------------------------
st.set_page_config(
    page_title="🦇 Batman Crime Predictor",
    layout="wide",
    initial_sidebar_state="auto"
)

st.markdown(
    """
    <style>
    :root{
      --bat-yellow: #f1c40f;
      --bg-dark: #0d0d0d;
      --panel-dark: #1a1a1a;
      --text-light: #f5f5f5;
      --bat-orange-dark: #e67e22;
      --bat-orange-light: #f39c12;
    }

    /* App background & text */
    html, body, .stApp {
      background-color: var(--bg-dark) !important;
      color: var(--text-light) !important;
    }

    /* Headings */
    h1, h2, h3, h4,
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4,
    .css-1v3fvcr h1, .css-10trblm h1 {
      color: var(--bat-yellow) !important;
      text-shadow: 0 0 6px rgba(241,196,15,0.12);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
      background-color: var(--panel-dark) !important;
      color: var(--bat-yellow) !important;
    }
    section[data-testid="stSidebar"] * {
      color: var(--bat-yellow) !important;
    }

    /* Buttons */
    div.stButton > button, button[kind="secondary"], button {
      background-color: var(--bat-yellow) !important;
      color: #000000 !important;
      border: 1px solid #000000 !important;
      font-weight: 600 !important;
      box-shadow: 0 2px 0 rgba(0,0,0,0.6) !important;
    }
    .stButton>button:hover {
      background-color: #d4ac0d !important;
      color: #000 !important;
    }

    /* DataFrames */
    .stDataFrame, .dataframe, .element-container {
      background-color: transparent !important;
      color: var(--text-light) !important;
    }

    /* Metric values */
    [data-testid="stMetricValue"] {
      color: var(--bat-yellow) !important;
    }

    /* Cards / Panels */
    .css-1d391kg, .css-1vsu8ta, .css-11h0k4r, .css-18e3th9 {
      background-color: var(--panel-dark) !important;
      color: var(--text-light) !important;
    }

    /* Plotly bg fix */
    .js-plotly-plot .plotly, .plot-container {
      background-color: transparent !important;
    }

    /* Links */
    a, a:link, a:visited {
      color: var(--bat-yellow) !important;
    }

    /* Progress bars */
    .prob-bar {
    height: 22px;
    border-radius: 6px;
    margin-bottom: 8px;
    background-color: #333;
    overflow: hidden;
    }

    .prob-fill-violent {
    background: linear-gradient(90deg, #e65100, #ff6f00); /* برتقالي غامق → أفتح */
    height: 100%;
    text-align: right;
    padding-right: 8px;
    color: #fff;
    font-weight: bold;
    }

    .prob-fill-nonviolent {
    background: linear-gradient(90deg, #ff9800, #ffcc80); /* برتقالي أفتح */
    height: 100%;
    text-align: right;
    padding-right: 8px;
    color: #000;
    font-weight: bold;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------
# Helpers: load artifacts safely
# ----------------------------
FILES_DIR = Path("Files")

@st.cache_resource
def load_pickle(fname):
    path = FILES_DIR / fname
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_data(sample_size=50000):
    df = pd.read_csv(
        "Sample_200k.csv",
        parse_dates=["DATE OCC"],
    )

    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
    return df

# --------------------
# Load artifacts
# --------------------
le_crm_cd_desc = load_pickle("Crm Cd Desc_label_encoder.pkl")
le_vict_descent = load_pickle("Vict Descent_label_encoder.pkl")
le_weapon_desc = load_pickle("Weapon Desc_label_encoder.pkl")
le_status_desc = load_pickle("Status Desc_label_encoder.pkl")
le_y = load_pickle("le_y.pkl")

yeo_johnson = load_pickle("yeo-johnson.pkl")
std_scaler = load_pickle("StandardScaler.pkl")
std2_scaler = load_pickle("StandardScaler2.pkl")
robust_scaler = load_pickle("RobustScaler.pkl")

# load models
available_models = {}
if FILES_DIR.exists():
    for p in FILES_DIR.glob("*.pkl"):
        name = p.stem
        if any(skip in name for skip in [
            "encoder", "Scaler", "scaler", "imputer", "yeo", "le_y",
            "X_train", "y_train", "X_test", "y_test", "results_dict"
        ]):
            continue
        available_models[name] = str(p)

# mapping for Vict Sex
VICT_SEX_MAP = {'M':0,'F':1,'H':2,'X':3}

# final feature list
MODEL_FEATURES = [
    "Crm Cd 1",
    "Crm Cd Desc",
    "Weapon Used Cd",
    "Premis Cd",
    "Weapon Desc",
    "Vict Age",
    "Vict Descent",
    "Vict Sex",
    "Status Desc"
]

def encode_and_transform(input_df: pd.DataFrame):
    df = input_df.copy()

    if le_crm_cd_desc is not None:
        df["Crm Cd Desc"] = le_crm_cd_desc.transform(df["Crm Cd Desc"].astype(str))
    else:
        df["Crm Cd Desc"], _ = pd.factorize(df["Crm Cd Desc"].astype(str))

    if le_vict_descent is not None:
        df["Vict Descent"] = le_vict_descent.transform(df["Vict Descent"].astype(str))
    else:
        df["Vict Descent"], _ = pd.factorize(df["Vict Descent"].astype(str))

    if le_weapon_desc is not None:
        df["Weapon Desc"] = le_weapon_desc.transform(df["Weapon Desc"].astype(str))
    else:
        df["Weapon Desc"], _ = pd.factorize(df["Weapon Desc"].astype(str))

    if le_status_desc is not None:
        df["Status Desc"] = le_status_desc.transform(df["Status Desc"].astype(str))
    else:
        df["Status Desc"], _ = pd.factorize(df["Status Desc"].astype(str))

    df["Vict Sex"] = df["Vict Sex"].map(lambda v: VICT_SEX_MAP.get(v, v) if isinstance(v, str) else v)
    df["Vict Sex"] = pd.to_numeric(df["Vict Sex"], errors="coerce").fillna(0)

    if std_scaler is not None:
        df[["Crm Cd Desc","Vict Descent"]] = std_scaler.transform(df[["Crm Cd Desc","Vict Descent"]])
    for c in ["Crm Cd 1","Premis Cd"]:
        df[c] = np.log1p(pd.to_numeric(df[c], errors="coerce").fillna(0))
    if std2_scaler is not None:
        df[["Crm Cd 1","Premis Cd"]] = std2_scaler.transform(df[["Crm Cd 1","Premis Cd"]])
    for c in ["Vict Age","Weapon Used Cd"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    if robust_scaler is not None:
        df[["Vict Age","Weapon Used Cd"]] = robust_scaler.transform(df[["Vict Age","Weapon Used Cd"]])

    if yeo_johnson is not None:
        apply_cols = [c for c in ["Weapon Desc","Status Desc","Vict Sex"] if c in df.columns]
        if apply_cols:
            df[apply_cols] = yeo_johnson.transform(df[apply_cols])

    return df[MODEL_FEATURES]

# ----------------------------
# Sidebar navigation
# ----------------------------
st.sidebar.title("Gotham Navigator")
page = st.sidebar.selectbox("Navigate", ["Home", "Predict", "Visualizations", "Model Comparison", "About"])


# ----------------------------
# HOME
# ----------------------------
if page == "Home":
    st.title("Los Angeles Crime Explorer")
    st.markdown("**Dataset**: LAPD — Crime incidents (2020 - present). This app lets you explore, visualize, and predict whether a given incident is Violent or Non-Violent.")
    st.markdown("---")
    st.header("Quick Actions")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Load sample data**")
        if st.button("Load sample (5k rows)"):
            if Path("Sample_200k.csv").exists():
                df_sample = pd.read_csv("Sample_200k.csv", nrows=5000)
                st.dataframe(df_sample.head())
            else:
                st.warning("Dataset CSV not found in project folder.")

    with col2:
        st.markdown("**Available models**")
        if available_models:
            st.write(list(available_models.keys()))
        else:
            st.warning("No ML models (.pkl) found in Files/.")

    st.markdown("---")
    st.markdown("**How to use**")
    st.markdown("""
    1. Go to **Predict** to input an incident's features and get a prediction (**Violent** / **Non-Violent**).  
    2. Go to **Visualizations** to explore the crime map and time heatmaps.  
    3. Use **Model Comparison** to inspect saved metrics and confusion matrices.
    """)
    st.caption("Tip: for best results, use the same encoders/scalers that were used during training (stored in Files/).")

# ----------------------------
# PREDICT PAGE
# ----------------------------
elif page == "Predict":
    st.title("Predict Crime Severity (Part 1 vs Part 2)")
    st.markdown("Fill the fields below (values should match training encodings where applicable).")

    model_choice = st.selectbox("Choose model for prediction", options=list(available_models.keys()))
    model_path = available_models.get(model_choice)
    if model_path is None:
        st.error("Model file not found.")
        st.stop()

    model = load_pickle(Path(model_path).name)
    if model is None:
        st.error("Could not load model pickle. Check Files/ folder.")
        st.stop()

    le_y = load_pickle("le_y.pkl")

    # Build input form
    with st.form("predict_form"):
        st.subheader("Incident features")
        col1, col2, col3 = st.columns(3)

        with col1:
            crm_cd1 = st.number_input("Crm Cd 1 (numeric code)", value=200.0, step=1.0)
            crm_cd_desc = st.text_input("Crm Cd Desc", value="THEFT OF IDENTITY")
            weapon_used_cd = st.number_input("Weapon Used Cd (numeric code)", value=0.0, step=1.0)
        with col2:
            premis_cd = st.number_input("Premis Cd (location code)", value=400.0, step=1.0)
            weapon_desc = st.text_input("Weapon Desc", value="No Weapon")
            vict_age = st.number_input("Victim Age", value=30.0, step=1.0, min_value=0.0)
        with col3:
            vict_descent = st.selectbox("Victim Descent", options=["H","W","B","O","A","I","U","X","Z"], index=0)
            vict_sex = st.selectbox("Vict Sex", options=["M","F","H","X"], index=0)
            status_desc = st.text_input("Status Desc", value="Invest Cont")

        submit = st.form_submit_button("Predict")

    if submit:
        # assemble single-row df
        sample = pd.DataFrame([{
            "Crm Cd 1": crm_cd1,
            "Crm Cd Desc": crm_cd_desc,
            "Weapon Used Cd": weapon_used_cd,
            "Premis Cd": premis_cd,
            "Weapon Desc": weapon_desc,
            "Vict Age": vict_age,
            "Vict Descent": vict_descent,
            "Vict Sex": vict_sex,
            "Status Desc": status_desc
        }])
    
        # apply encoding + transforms
        try:
            X_sample = encode_and_transform(sample)
        except Exception as e:
            st.error(f"Preprocessing failed: {e}")
            st.stop()
    
        # predict
        pred = model.predict(X_sample)[0]
        original_class = le_y.inverse_transform([pred])[0]  
        pred_label = "Violent" if original_class == 1 else "Non-Violent"
    
        # get probabilities
        prob_violent, prob_non_violent = 0, 0
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X_sample)[0]
            if len(prob) == 2:
                prob_violent, prob_non_violent = prob[0], prob[1]
    
        # اختيار اللون حسب النتيجة
        title_color = "#e65100" if pred_label == "Violent" else "#f39c12"
    
        # custom styled output
        st.markdown(f"""
            <h3 style="color:{title_color};">Predicted: {pred_label}</h3>
    
            <div class="prob-bar">
                <div class="prob-fill-violent" style="width:{prob_violent*100:.2f}%; ">
                    Violent {prob_violent:.3f}
                </div>
            </div>
    
            <div class="prob-bar">
                <div class="prob-fill-nonviolent" style="width:{prob_non_violent*100:.2f}%; ">
                    Non-Violent {prob_non_violent:.3f}
                </div>
            </div>
        """, unsafe_allow_html=True)
    


# ----------------------------
# VISUALIZATIONS
# ----------------------------

elif page == "Visualizations":
    st.title("Visual Analytics")
    st.markdown("Interactive visualizations for exploration. (Cached & sampled at 50k rows for speed.)")

    tab1, tab2, tab3 = st.tabs(["Crime Map", "Time Heatmap", "Charts"])

    # --------------------- MAP ---------------------
    with tab1:
        st.header("Crime Map")
        if Path("crime_map.html").exists():
            with open("crime_map.html", "r", encoding="utf-8") as f:
                html_data = f.read()
            components.html(html_data, height=600)  
        else:
            st.warning("crime_map.html not found in app directory.")

    # --------------------- HEATMAP ---------------------
    with tab2:
        st.header("Heatmap: Day of Week × Hour of Day")
        if Path("Sample_200k.csv").exists():
            df_full = load_data()

            if "TIME OCC" in df_full.columns:
                df_full["Hour OCC"] = df_full["TIME OCC"].apply(
                    lambda x: int(str(int(x)).zfill(4)[:2]) if pd.notnull(x) else 0
                )
            else:
                df_full["Hour OCC"] = 0

            df_full["day_of_week OCC"] = df_full["DATE OCC"].dt.dayofweek

            pivot = pd.pivot_table(
                df_full,
                values="DR_NO", 
                index="day_of_week OCC",
                columns="Hour OCC",
                aggfunc="count",
                fill_value=0
            )
            pivot.index = pivot.index.map({0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"})
            pivot_long = pivot.reset_index().melt(
                id_vars="day_of_week OCC",
                var_name="Hour",
                value_name="Crime_Count"
            )

            fig = px.density_heatmap(
                pivot_long,
                x="Hour",
                y="day_of_week OCC",
                z="Crime_Count",
                color_continuous_scale="YlOrRd",
                labels={"Crime_Count": "Number of Crimes"}
            )
            fig.update_layout(width=900, height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("CSV not found — heatmap unavailable.")

    # --------------------- CHARTS ---------------------
    with tab3:
        st.header("Key Charts")
        st.markdown("Monthly trend and top crime types.")

        if Path("Sample_200k.csv").exists():
            df_full = load_data()

            # Monthly trend
            df_full["year_month"] = df_full["DATE OCC"].dt.to_period("M").dt.to_timestamp()
            crimes_monthly = df_full.groupby("year_month").size().reset_index(name="count")
            fig = px.line(
                crimes_monthly,
                x="year_month",
                y="count",
                markers=True,
                title="Monthly Crime Trend",
                color_discrete_sequence=["#d73027"]
            )
            fig.update_layout(
                plot_bgcolor="#fff7cc",
                paper_bgcolor="#fff2a8",
                width=900,
                height=450
            )
            st.plotly_chart(fig, use_container_width=True)

            # Top crime types
            if "Crm Cd Desc" in df_full.columns:
                top = df_full["Crm Cd Desc"].value_counts().nlargest(12)
                fig2 = px.bar(
                    x=top.values,
                    y=top.index,
                    orientation='h',
                    title="Top Crime Types",
                    color=top.values,
                    color_continuous_scale="OrRd"
                )
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("Column 'Crm Cd Desc' not found in CSV.")
        else:
            st.info("CSV not available — charts not shown.")
# ----------------------------
# MODEL COMPARISON
# ----------------------------
elif page == "Model Comparison":
    st.title("Model Comparison")
    st.markdown("Compare metrics, confusion matrices, and classification reports for saved models.")

    results_path = FILES_DIR / "results_dict.pkl"
    if not results_path.exists():
        st.warning("No results file found. Please generate 'results_dict.pkl' in your notebook first.")
        st.stop()

    with open(results_path, "rb") as f:
        results_dict = pickle.load(f)

    model_names = list(results_dict.keys())
    selected_model = st.selectbox("Select model to inspect", options=model_names)

    if selected_model:
        res = results_dict[selected_model]

        # Metrics table
        st.subheader("Metrics")
        metrics_df = pd.DataFrame([res['metrics']])
        st.dataframe(metrics_df)

        # Confusion matrix
        st.subheader("Confusion Matrix")
        cm = res['confusion_matrix']
        class_names = res.get('class_names', [0,1])
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        # Classification report
        st.subheader("Classification Report")
        cls_report = res['classification_report']
        cls_df = pd.DataFrame(cls_report).transpose()
        st.dataframe(cls_df)

# ----------------------------
# ABOUT
# ----------------------------
elif page == "About":
    st.title("About this App")
    st.markdown("""
    **LA Crime Explorer** — a Streamlit app to explore Los Angeles Police Department crime data, visualize hotspots, and run a classifier for Part 1 vs Part 2 crimes.
    
    * Built from my notebook’s preprocessing pipeline and saved artifacts in `Files/`.
    * Input page runs the exact same transforms (label encoders, scalers) before using saved models.
    * Visualizations: crime map, heatmap (day×hour), monthly trends, top crime types.
    """)
    st.markdown("**How to run**")
    st.code("""
    pip install -r requirements.txt
    streamlit run app.py
    """)
    st.markdown("**Requirements**: streamlit, pandas, numpy, matplotlib, seaborn, plotly, scikit-learn, xgboost, lightgbm, catboost, pickle, pathlib")

# End of app
