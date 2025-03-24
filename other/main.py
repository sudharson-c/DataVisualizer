import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Classify Data Types
def classify_columns(df):
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
        elif unique_ratio < 0.1 and avg_length < 50:
            categorical_cols.append(col)
        else:
            text_cols.append(col)

    for col in numerical_cols[:]:
        if df[col].nunique() < 50 and 'id' not in col.lower():
            numerical_cols.remove(col)
            categorical_cols.append(col)

    return numerical_cols, categorical_cols, text_cols, boolean_cols, sparse_cols

# Step 2: Missing Value Analysis
def analyze_missing_values(df):
    missing_counts = df.isna().sum()
    total_rows = len(df)
    missing_info = [(col, count, f"{(count / total_rows) * 100:.1f}%") 
                    for col, count in missing_counts.items() if count > 0]
    return missing_info

# Step 3: Data Insights (Distribution, Redundancy, Outliers)
def analyze_data_insights(df, corr_threshold=0.95):
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    insights = {}

    # Distribution & Outliers
    outliers = {}
    for col in numerical_cols:
        if df[col].std() > 0:  # Skip constant columns
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

# Step 4: Generate Summary
def generate_summary(df, num_cols, cat_cols, text_cols, bool_cols, sparse_cols, missing_info, insights):
    total_cols = df.shape[1]
    missing_cols = len(missing_info)
    data_health = 100 * (1 - missing_cols / total_cols - len(sparse_cols) / total_cols)

    summary = "### 📊 Data Analysis Report\n"
    summary += f"📂 **Dataset Size:** {df.shape[0]} rows, {df.shape[1]} columns\n"
    summary += f"🌟 **Data Health Score:** {data_health:.0f}/100 (based on missingness)\n\n"

    summary += f"📊 **Numerical Columns:** {len(num_cols)}\n"
    summary += f"🔠 **Categorical Columns:** {len(cat_cols)}\n"
    summary += f"📝 **Text Columns:** {len(text_cols)}\n"
    summary += f"✅ **Boolean Columns:** {len(bool_cols)}\n"
    summary += f"⚠ **Sparse Columns:** {len(sparse_cols)}\n\n"

    if missing_info:
        summary += f"❓ **Missing Data:** {missing_cols} columns have missing values\n"
    if insights["redundant_cols"]:
        summary += f"🔗 **Redundant Columns:** {insights['redundant_cols']}\n"
    if any(insights["outliers"].values()):
        summary += f"⚠ **Outliers Detected:** In {sum(v > 0 for v in insights['outliers'].values())} columns\n"

    summary += "\n### 💡 Tips for Next Steps\n"
    if sparse_cols:
        summary += f"- Consider **dropping sparse columns**: {sparse_cols}\n"
    if missing_info:
        summary += "- **Fill or drop** columns with missing values (see Step 2)\n"
    if insights["redundant_cols"]:
        summary += "- **Remove redundant columns** to simplify analysis\n"
    if any(insights["outliers"].values()):
        summary += "- **Check outliers** in numerical columns (see Step 3)\n"
    summary += "- Explore categorical/text data with charts or further cleaning\n"

    return summary

# Streamlit UI
st.title("📊 Easy Data Profiler")
st.write("Upload your dataset to get a quick analysis and tips!")

uploaded_file = st.file_uploader("📂 Upload CSV, Excel, or JSON", type=["csv", "xlsx", "json"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_json(uploaded_file)

        st.write("### 🔍 Data Preview")
        st.dataframe(df.head())

        # Step 1: Column Types
        with st.expander("1. What’s in My Data?"):
            num_cols, cat_cols, text_cols, bool_cols, sparse_cols = classify_columns(df)
            st.write(f"📊 **Numbers:** {', '.join(num_cols) or 'None'}")
            st.write(f"🔠 **Categories:** {', '.join(cat_cols) or 'None'}")
            st.write(f"📝 **Text:** {', '.join(text_cols) or 'None'}")
            st.write(f"✅ **Yes/No:** {', '.join(bool_cols) or 'None'}")
            st.write(f"⚠ **Mostly Empty:** {', '.join(sparse_cols) or 'None'}")
            st.info("This step identifies the types of data in your columns.")

        # Step 2: Missing Values
        with st.expander("2. Missing Data Check"):
            missing_info = analyze_missing_values(df)
            if missing_info:
                st.dataframe(pd.DataFrame(missing_info, columns=["Column", "Missing Count", "Missing %"]))
                fig, ax = plt.subplots()
                sns.heatmap(df.isna(), cbar=False, cmap="binary", ax=ax)
                st.pyplot(fig)
                st.write("**Tip:** Dark lines show missing data. Fill small gaps, drop columns with lots of gaps.")
            else:
                st.success("No missing data found!")

        # Step 3: Data Insights
        with st.expander("3. Numbers & Issues"):
            insights = analyze_data_insights(df)
            num_cols = df.select_dtypes(include=['int64', 'float64']).columns

            if num_cols.any():
                st.write("#### 📈 Number Patterns")
                for col in num_cols:
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
                    sns.histplot(df[col].dropna(), bins=20, ax=ax1)
                    ax1.set_title(f"{col} Distribution")
                    sns.boxplot(x=df[col].dropna(), ax=ax2)
                    ax2.set_title(f"{col} Outliers")
                    st.pyplot(fig)
                    if insights["outliers"][col] > 0:
                        st.write(f"⚠ {col} has {insights['outliers'][col]} potential outliers")

                if insights["redundant_cols"]:
                    st.warning(f"🔗 These columns might be duplicates: {insights['redundant_cols']}")
            else:
                st.write("No numerical data to analyze.")

            if cat_cols:
                st.write("#### 🔠 Category Counts")
                for col in cat_cols[:5]:  # Limit to 5 for simplicity
                    fig, ax = plt.subplots()
                    df[col].value_counts().plot.bar(ax=ax)
                    plt.title(f"{col} Breakdown")
                    st.pyplot(fig)

        # Step 4: Summary
        with st.expander("4. Summary & Tips"):
            summary = generate_summary(df, num_cols, cat_cols, text_cols, bool_cols, sparse_cols, missing_info, insights)
            st.markdown(summary)
            st.download_button("📥 Download Report", summary, "data_report.txt")

    except Exception as e:
        st.error(f"❌ Error: {e}")