
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Classify Data Types
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

# Missing Value Analysis
def analyze_missing_values(df):
    missing_counts = df.isna().sum()
    total_rows = len(df)
    missing_info = [(col, count, f"{(count / total_rows) * 100:.1f}%") 
                    for col, count in missing_counts.items() if count > 0]
    return missing_info

# Data Insights (Distribution, Redundancy, Outliers)
def analyze_data_insights(df, corr_threshold=0.95):
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    insights = {}

    # Distribution Stats
    dist_stats = {}
    for col in numerical_cols:
        if df[col].std() > 0:
            dist_stats[col] = {
                "mean": df[col].mean(),
                "std": df[col].std(),
                "min": df[col].min(),
                "max": df[col].max()
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

    return insights

# Generate Summary
def generate_summary(df, num_cols, cat_cols, text_cols, bool_cols, sparse_cols, missing_info, insights):
    total_cols = df.shape[1]
    missing_cols = len(missing_info)
    data_health = 100 * (1 - missing_cols / total_cols - len(sparse_cols) / total_cols)

    summary = "### 📊 Data Profile Report\n"
    summary += f"📂 **Dataset Size:** {df.shape[0]} rows, {df.shape[1]} columns\n"
    summary += f"🌟 **Data Health Score:** {data_health:.0f}/100 (lower if missing or sparse data)\n\n"

    summary += f"📊 **Numerical Columns:** {len(num_cols)} (e.g., {', '.join(num_cols[:3]) + '...' if num_cols else 'None'})\n"
    summary += f"🔠 **Categorical Columns:** {len(cat_cols)} (e.g., {', '.join(cat_cols[:3]) + '...' if cat_cols else 'None'})\n"
    summary += f"📝 **Text Columns:** {len(text_cols)} (e.g., {', '.join(text_cols[:3]) + '...' if text_cols else 'None'})\n"
    summary += f"✅ **Boolean Columns:** {len(bool_cols)} (e.g., {', '.join(bool_cols[:3]) + '...' if bool_cols else 'None'})\n"
    summary += f"⚠ **Sparse Columns:** {len(sparse_cols)} (e.g., {', '.join(sparse_cols[:3]) + '...' if sparse_cols else 'None'})\n\n"

    if missing_info:
        summary += f"❓ **Missing Data:** {missing_cols} columns have gaps\n"
    if insights["redundant_cols"]:
        summary += f"🔗 **Redundant Columns:** {insights['redundant_cols']}\n"
    if any(insights["outliers"].values()):
        summary += f"⚠ **Outliers Detected:** In {sum(v > 0 for v in insights['outliers'].values())} numerical columns\n"

    summary += "\n### 💡 Next Steps for Analysis\n"
    if sparse_cols:
        summary += f"- **Drop sparse columns** like {sparse_cols[:3]} if not critical\n"
    if missing_info:
        summary += "- **Handle missing values** (fill with average or remove rows/columns)\n"
    if insights["redundant_cols"]:
        summary += "- **Remove redundant columns** to avoid duplication\n"
    if any(insights["outliers"].values()):
        summary += "- **Investigate outliers**—they might skew results\n"
    summary += "- **Explore categorical data** with charts to spot trends\n"

    return summary


st.set_page_config(page_title="Data Profiler", layout="wide")
st.title("📊 Data Profiler")
st.markdown("Upload your dataset to explore its structure, spot issues, and get analysis tips!")

with st.sidebar:
    st.header("⚙️ Settings")
    cat_threshold = st.slider("Categorical Threshold", 0.05, 0.5, 0.1, help="Lower values = stricter categorical detection")
    corr_threshold = st.slider("Correlation Threshold", 0.8, 1.0, 0.95, help="Higher values = fewer redundant columns flagged")
    plot_style = st.selectbox("Plot Style", ["Histogram", "Box Plot", "Both"], help="Choose how to visualize numerical data")
    bins = st.slider("Histogram Bins", 10, 50, 20, help="Number of bars in histograms")


uploaded_file = st.file_uploader("📂 Upload Your Dataset (CSV, Excel, JSON)", type=["csv", "xlsx", "json"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_json(uploaded_file)

        tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Missing Data", "Data Insights", "Summary"])

        with tab1:
            st.subheader("1. Dataset Overview")
            st.dataframe(df.head(), use_container_width=True)
            num_cols, cat_cols, text_cols, bool_cols, sparse_cols = classify_columns(df, cat_threshold)
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Column Types:**")
                st.write(f"📊 Numerical: {', '.join(num_cols) or 'None'}")
                st.write(f"🔠 Categorical: {', '.join(cat_cols) or 'None'}")
            with col2:
                st.write(f"📝 Text: {', '.join(text_cols) or 'None'}")
                st.write(f"✅ Boolean: {', '.join(bool_cols) or 'None'}")
                st.write(f"⚠ Sparse: {', '.join(sparse_cols) or 'None'}")
            st.write(df.describe())
            

            st.info("**What this means:** Numerical columns have numbers, categorical have few unique values, text is longer strings, boolean is yes/no, and sparse is mostly empty.")

        with tab2:
            st.subheader("2. Missing Data Analysis")
            missing_info = analyze_missing_values(df)
            if missing_info:
                st.dataframe(pd.DataFrame(missing_info, columns=["Column", "Missing Count", "Missing %"]), use_container_width=True)
                st.write("**Visualize Missingness:**")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(df.isna(), cbar=False, cmap="binary", ax=ax)
                st.pyplot(fig)
                st.info("**Tip:** Dark areas show missing values. Fill small gaps with averages or drop columns with too many gaps.")
            else:
                st.success("✅ No missing data found!")

        with tab3:
            st.subheader("3. Data Insights")
            insights = analyze_data_insights(df, corr_threshold)

            # Numerical Insights
            if num_cols:
                st.write("**Numerical Data:**")
                st.dataframe(pd.DataFrame(insights["dist_stats"]).T, use_container_width=True)
                for col in num_cols:
                    st.write(f"**{col} Visualization:**")
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
                        st.warning(f"⚠ {col} has {insights['outliers'][col]} outliers (values far from the norm)")

                if insights["redundant_cols"]:
                    st.warning(f"🔗 Redundant Columns: {insights['redundant_cols']} (highly similar to others)")

            # Categorical Insights
            if cat_cols:
                st.write("**Categorical Data:**")
                for col in cat_cols[:5]:  # Limit to 5
                    fig, ax = plt.subplots()
                    df[col].value_counts().plot.bar(ax=ax)
                    ax.set_title(f"{col} Breakdown")
                    st.pyplot(fig)

        with tab4:
            st.subheader("4. Summary & Recommendations")
            summary = generate_summary(df, num_cols, cat_cols, text_cols, bool_cols, sparse_cols, missing_info, insights)
            st.markdown(summary)
            st.download_button("📥 Download Report", summary, "data_profile_report.txt", help="Save this summary as a text file")

        st.subheader("Export Cleaned Data")
        cleaned_df = df.drop(columns=sparse_cols + insights["redundant_cols"])
        csv = cleaned_df.to_csv(index=False)
        st.download_button("📤 Download Cleaned CSV", csv, "cleaned_data.csv", help="Dataset with sparse/redundant columns removed")

    except Exception as e:
        st.error(f"❌ Error: {e}")