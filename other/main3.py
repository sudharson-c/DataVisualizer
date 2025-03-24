import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Step 1: Classify Data Types
def classify_columns(df, cat_threshold=0.1, text_length_threshold=50):
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = []
    text_cols = []
    boolean_cols = []
    sparse_cols = [col for col in df.columns if df[col].isna().mean() > 0.9]

    for col in df.select_dtypes(include=['object', 'bool']).columns:
        if col in sparse_cols:
            continue
        unique_ratio = df[col].nunique() / len(df[col].dropna())
        avg_length = df[col].dropna().str.len().mean() if df[col].dtype == 'object' else 0

        if df[col].dropna().isin([0, 1, True, False, 'yes', 'no', 'true', 'false']).all():
            boolean_cols.append(col)
        elif unique_ratio < cat_threshold and avg_length < text_length_threshold:
            categorical_cols.append(col)
        else:
            text_cols.append(col)

    for col in numerical_cols[:]:
        if df[col].nunique() < 50 and 'id' not in col.lower():
            numerical_cols.remove(col)
            categorical_cols.append(col)

    return numerical_cols, categorical_cols, text_cols, boolean_cols, sparse_cols

# Step 2: Cleaning Checks
def cleaning_checks(df):
    checks = {}
    checks["missing"] = [(col, count, f"{(count / len(df)) * 100:.1f}%") 
                         for col, count in df.isna().sum().items() if count > 0]
    checks["duplicates"] = df.duplicated().sum()
    checks["inconsistencies"] = {}
    for col in df.select_dtypes(include=['object']).columns:
        if col not in checks["missing"]:
            unique_formats = df[col].dropna().str.match(r'^\d{2}/\d{2}/\d{2}$').nunique() > 1  # Simple date format check
            if unique_formats:
                checks["inconsistencies"][col] = "Possible mixed formats (e.g., dates)"

    return checks

# Step 3: Data Insights
def analyze_data_insights(df, corr_threshold=0.95):
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    insights = {}

    # Stats
    dist_stats = {}
    for col in numerical_cols:
        if df[col].std() > 0:
            dist_stats[col] = {
                "mean": df[col].mean(),
                "std": df[col].std(),
                "min": df[col].min(),
                "max": df[col].max(),
                "mode" : df[col].mode()[0],
                "skewness": df[col].skew(),
                "kurtosis": df[col].kurt(),
            }
    insights["dist_stats"] = dist_stats

    # Outliers
    outliers = {}
    for col in numerical_cols:
        if df[col].std() > 0:
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            lower_bound, upper_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            outliers[col] = outlier_count
    insights["outliers"] = outliers

    # Redundancy
    if len(numerical_cols) > 1:
        corr_matrix = df[numerical_cols].corr().abs()
        redundant_cols = set()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if corr_matrix.iloc[i, j] > corr_threshold:
                    redundant_cols.add(corr_matrix.columns[i])
        insights["redundant_cols"] = list(redundant_cols)
    else:
        insights["redundant_cols"] = []

    # Validation
    insights["validation"] = {}
    for col in numerical_cols:
        if df[col].min() < 0 and "age" in col.lower():  # Example sanity check
            insights["validation"][col] = "Negative values detected where unexpected"

    return insights

# Step 4: Summary
def generate_summary(df, num_cols, cat_cols, text_cols, bool_cols, sparse_cols, checks, insights):
    total_cols = df.shape[1]
    missing_cols = len(checks["missing"])
    data_health = 100 * (1 - missing_cols / total_cols - len(sparse_cols) / total_cols - checks["duplicates"] / len(df))

    summary = "### 📊 Data Profile Report\n"
    summary += f"📂 **Dataset Size:** {df.shape[0]} rows, {df.shape[1]} columns\n"
    summary += f"🌟 **Data Health Score:** {data_health:.0f}/100 (penalized for missing, sparse, or duplicate data)\n\n"

    summary += f"📊 **Numerical Columns:** {len(num_cols)} (e.g., {', '.join(num_cols[:3]) + '...' if num_cols else 'None'})\n"
    summary += f"🔠 **Categorical Columns:** {len(cat_cols)} (e.g., {', '.join(cat_cols[:3]) + '...' if cat_cols else 'None'})\n"
    summary += f"📝 **Text Columns:** {len(text_cols)} (e.g., {', '.join(text_cols[:3]) + '...' if text_cols else 'None'})\n"
    summary += f"✅ **Boolean Columns:** {len(bool_cols)} (e.g., {', '.join(bool_cols[:3]) + '...' if bool_cols else 'None'})\n"
    summary += f"⚠ **Sparse Columns:** {len(sparse_cols)} (e.g., {', '.join(sparse_cols[:3]) + '...' if sparse_cols else 'None'})\n\n"

    if checks["missing"]:
        summary += f"❓ **Missing Data:** {missing_cols} columns\n"
    if checks["duplicates"]:
        summary += f"🔍 **Duplicates:** {checks['duplicates']} rows\n"
    if checks["inconsistencies"]:
        summary += f"⚠ **Inconsistencies:** {list(checks['inconsistencies'].keys())}\n"
    if insights["redundant_cols"]:
        summary += f"🔗 **Redundant Columns:** {insights['redundant_cols']}\n"
    if any(insights["outliers"].values()):
        summary += f"⚠ **Outliers:** In {sum(v > 0 for v in insights['outliers'].values())} columns\n"
    if insights["validation"]:
        summary += f"🚨 **Validation Issues:** {list(insights['validation'].values())}\n"

    summary += "\n### 💡 Next Steps\n"
    if sparse_cols:
        summary += f"- **Drop sparse columns**: {sparse_cols[:3]}\n"
    if checks["missing"]:
        summary += "- **Handle missing values**: Fill or drop\n"
    if checks["duplicates"]:
        summary += "- **Remove duplicates** to clean data\n"
    if insights["redundant_cols"]:
        summary += "- **Drop redundant columns**\n"
    if any(insights["outliers"].values()):
        summary += "- **Check outliers** for errors or insights\n"

    return summary

# Streamlit UI
st.set_page_config(page_title="Data Profiler", layout="wide")
st.title("📊 Your Data Analyser ")
st.markdown("Analyze your dataset with ease—get insights, spot issues, and prep for analysis!")

# Sidebar
with st.sidebar:
    st.header("⚙️ Analysis Settings")
    goal = st.selectbox("What’s your goal?", ["Explore Trends", "Clean Data", "Prepare for Modeling"])
    cat_threshold = st.slider("Categorical Threshold", 0.05, 0.5, 0.1)
    corr_threshold = st.slider("Correlation Threshold", 0.8, 1.0, 0.95)
    plot_style = st.selectbox("Plot Style", ["Histogram", "Box Plot", "Both"])
    bins = st.slider("Histogram Bins", 10, 50, 20)

uploaded_file = st.file_uploader("📂 Upload Dataset (CSV, Excel, JSON)", type=["csv", "xlsx", "json"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_json(uploaded_file)

        tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Cleaning", "Insights", "Summary"])

        with tab1:
            st.subheader("1. Dataset Overview")
            st.dataframe(df.head(), use_container_width=True)
            num_cols, cat_cols, text_cols, bool_cols, sparse_cols = classify_columns(df, cat_threshold)
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"📊 Numerical ({len(num_cols)}): {', '.join(num_cols) or 'None'}")
                st.write(f"🔠 Categorical ({len(cat_cols)}): {', '.join(cat_cols) or 'None'}")
            with col2:
                st.write(f"📝 Text ({len(text_cols)}): {', '.join(text_cols) or 'None'}")
                st.write(f"✅ Boolean ({len(bool_cols)}): {', '.join(bool_cols) or 'None'}")
                st.write(f"⚠ Sparse ({len(sparse_cols)}): {', '.join(sparse_cols) or 'None'}")
            st.info("This shows what types of data you have—numbers, categories, etc.")

        with tab2:
            st.subheader("2. Cleaning Checks")
            checks = cleaning_checks(df)
            if checks["missing"]:
                st.dataframe(pd.DataFrame(checks["missing"], columns=["Column", "Missing Count", "%"]), use_container_width=True)
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(df.isna(), cbar=False, cmap="binary", ax=ax)
                st.pyplot(fig)
            if checks["duplicates"]:
                st.warning(f"Found {checks['duplicates']} duplicate rows")
            if checks["inconsistencies"]:
                st.warning(f"Possible inconsistencies: {checks['inconsistencies']}")
            if not any(checks.values()):
                st.success("No major cleaning issues found!")

        with tab3:
            st.subheader("3. Data Insights")
            insights = analyze_data_insights(df, corr_threshold)

            if num_cols:
                st.write("**Numerical Insights:**")
                st.dataframe(pd.DataFrame(insights["dist_stats"]).T, use_container_width=True)
                for col in num_cols:
                    st.write(f"**{col}:**")
                    col1, col2 = st.columns(2)
                    if plot_style in ["Histogram", "Both"]:
                        with col1:
                            fig, ax = plt.subplots()
                            sns.histplot(df[col].dropna(), bins=bins, ax=ax)
                            ax.set_title(f"{col} Histogram")
                            st.pyplot(fig)
                    if plot_style in ["Box Plot", "Both"]:
                        with col2:
                            fig, ax = plt.subplots()
                            sns.boxplot(x=df[col].dropna(), ax=ax)
                            ax.set_title(f"{col} Box Plot")
                            st.pyplot(fig)
                    if insights["outliers"][col] > 0:
                        st.warning(f"{col}: {insights['outliers'][col]} outliers")

                if insights["redundant_cols"]:
                    st.warning(f"Redundant: {insights['redundant_cols']}")

            if cat_cols:
                st.write("**Categorical Insights:**")
                for col in cat_cols[:5]:
                    fig, ax = plt.subplots()
                    df[col].value_counts().plot.bar(ax=ax)
                    ax.set_title(f"{col} Breakdown")
                    st.pyplot(fig)

        with tab4:
            st.subheader("4. Summary & Next Steps")
            summary = generate_summary(df, num_cols, cat_cols, text_cols, bool_cols, sparse_cols, checks, insights)
            st.markdown(summary)
            st.download_button("📥 Download Report", summary, "data_profile_report.txt")

            cleaned_df = df.drop(columns=sparse_cols + insights["redundant_cols"]).drop_duplicates()
            csv = cleaned_df.to_csv(index=False)
            st.download_button("📤 Download Cleaned CSV", csv, "cleaned_data.csv")

    except Exception as e:
        st.error(f"❌ Error: {e}")