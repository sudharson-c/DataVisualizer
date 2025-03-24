import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Step 1: AI-Powered Data Classification
def classify_columns(df, cat_threshold=0.1, text_length_threshold=50):
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols, text_cols, boolean_cols, sparse_cols = [], [], [], []

    for col in df.columns:
        unique_ratio = df[col].nunique() / len(df[col].dropna())
        avg_length = df[col].astype(str).str.len().mean() if df[col].dtype == 'object' else 0

        if df[col].isna().mean() > 0.9:
            sparse_cols.append(col)
        elif df[col].dropna().isin([0, 1, True, False, 'yes', 'no']).all():
            boolean_cols.append(col)
        elif unique_ratio < cat_threshold and avg_length < text_length_threshold:
            categorical_cols.append(col)
        else:
            text_cols.append(col)

    # Identify misclassified numerical IDs
    for col in numerical_cols[:]:
        if df[col].nunique() < 50 and 'id' not in col.lower():
            numerical_cols.remove(col)
            categorical_cols.append(col)

    return numerical_cols, categorical_cols, text_cols, boolean_cols, sparse_cols

# Step 2: Data Cleaning Checks
def cleaning_checks(df):
    checks = {
        "missing": [(col, df[col].isna().sum(), f"{df[col].isna().mean() * 100:.1f}%")
                    for col in df.columns if df[col].isna().sum() > 0],
        "duplicates": df.duplicated().sum(),
        "inconsistencies": {col: "Mixed data formats detected" 
                            for col in df.select_dtypes(include=['object']).columns 
                            if df[col].dropna().str.match(r'^\d{2}/\d{2}/\d{2}$').nunique() > 1}
    }
    return checks

# Step 3: AI-Driven Data Insights
def analyze_data_insights(df, corr_threshold=0.95):
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    insights = {"dist_stats": {}, "outliers": {}, "redundant_cols": []}

    # Distribution Statistics
    for col in numerical_cols:
        insights["dist_stats"][col] = {
            "mean": df[col].mean(),
            "std": df[col].std(),
            "min": df[col].min(),
            "max": df[col].max(),
            "mode": df[col].mode()[0],
            "skewness": df[col].skew(),
            "kurtosis": df[col].kurt(),
        }

    # Outlier Detection (Z-score & IQR)
    for col in numerical_cols:
        if df[col].std() > 0:
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            iqr = stats.iqr(df[col].dropna())
            outliers = sum(z_scores > 3) + sum((df[col] < df[col].quantile(0.25) - 1.5 * iqr) |
                                               (df[col] > df[col].quantile(0.75) + 1.5 * iqr))
            insights["outliers"][col] = outliers

    # Redundant Column Detection
    if len(numerical_cols) > 1:
        corr_matrix = df[numerical_cols].corr().abs()
        redundant_cols = {corr_matrix.columns[i] for i in range(len(corr_matrix.columns)) for j in range(i)
                          if corr_matrix.iloc[i, j] > corr_threshold}
        insights["redundant_cols"] = list(redundant_cols)

    return insights

# Step 4: AI-Powered Summary & Recommendations
def generate_summary(df, num_cols, cat_cols, text_cols, bool_cols, sparse_cols, checks, insights):
    total_cols = df.shape[1]
    missing_cols = len(checks["missing"])
    data_health = 100 * (1 - missing_cols / total_cols - len(sparse_cols) / total_cols - checks["duplicates"] / len(df))

    summary = f"""
    ### 📊 AI-Powered Data Profile Report  
    - **Dataset Size:** {df.shape[0]} rows, {df.shape[1]} columns  
    - **Data Health Score:** {data_health:.0f}/100  

    **Column Breakdown**  
    - 📊 **Numerical:** {len(num_cols)}  
    - 🔠 **Categorical:** {len(cat_cols)}  
    - 📝 **Text:** {len(text_cols)}  
    - ✅ **Boolean:** {len(bool_cols)}  
    - ⚠ **Sparse:** {len(sparse_cols)}  

    **Key Issues Identified:**  
    - ❓ Missing Data: {missing_cols} columns  
    - 🔍 Duplicates: {checks['duplicates']} rows  
    - ⚠ Outliers Detected: {sum(insights['outliers'].values())} instances  
    - 🔗 Redundant Columns: {len(insights['redundant_cols'])}  
    """

    return summary

# Streamlit UI
st.set_page_config(page_title="AI Data Profiler", layout="wide")
st.title("📊 AI-Powered Data Profiler")

uploaded_file = st.file_uploader("📂 Upload Dataset (CSV, Excel, JSON)", type=["csv", "xlsx", "json"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else (
            pd.read_excel(uploaded_file) if uploaded_file.name.endswith(".xlsx") else pd.read_json(uploaded_file)
        )

        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Cleaning", "Insights", "Visuals", "Summary"])

        with tab1:
            st.subheader("1. Dataset Overview")
            num_cols, cat_cols, text_cols, bool_cols, sparse_cols = classify_columns(df)
            st.dataframe(df.head(), use_container_width=True)

        with tab2:
            st.subheader("2. Cleaning Checks")
            checks = cleaning_checks(df)
            if checks["missing"]:
                st.dataframe(pd.DataFrame(checks["missing"], columns=["Column", "Missing Count", "%"]), use_container_width=True)
            if checks["duplicates"]:
                st.warning(f"Found {checks['duplicates']} duplicate rows")

        with tab3:
            st.subheader("3. Data Insights")
            insights = analyze_data_insights(df)
            if insights["outliers"]:
                st.warning(f"Outliers detected in {sum(v > 0 for v in insights['outliers'].values())} columns")

        with tab4:
            st.subheader("4. Data Visualization")
            st.write("📉 **Correlation Heatmap**")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", linewidths=0.5, ax=ax)
            st.pyplot(fig)

            st.write("📊 **Histograms**")
            for col in num_cols:
                fig, ax = plt.subplots()
                sns.histplot(df[col], kde=True, bins=30, ax=ax)
                st.pyplot(fig)

        with tab5:
            st.subheader("5. Summary & Recommendations")
            summary = generate_summary(df, num_cols, cat_cols, text_cols, bool_cols, sparse_cols, checks, insights)
            st.markdown(summary)
            st.download_button("📥 Download Report", summary, "data_profile.txt")

    except Exception as e:
        st.error(f"❌ Error: {e}")
