import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

# Step 1: Classify Data Types
def classify_columns(df, categorical_threshold=0.1, text_length_threshold=50):
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = []
    text_cols = []
    boolean_cols = []
    sparse_cols = []

    for col in df.columns:
        if df[col].isna().mean() > 0.9:
            sparse_cols.append(col)
            continue

    for col in df.select_dtypes(include=['object', 'bool']).columns:
        if col in sparse_cols:
            continue
        unique_values = df[col].nunique()
        total_values = len(df[col].dropna())

        if df[col].dropna().isin([0, 1, True, False]).all():
            boolean_cols.append(col)
        elif df[col].dropna().str.match(r'^(yes|no|true|false)$', case=False).all():
            boolean_cols.append(col)
        elif unique_values > 0:
            ratio = unique_values / total_values
            avg_length = df[col].dropna().str.len().mean() if df[col].dtype == 'object' else 0
            if ratio < categorical_threshold and avg_length < text_length_threshold:
                categorical_cols.append(col)
            else:
                text_cols.append(col)

    for col in numerical_cols[:]:
        if df[col].nunique() < 50 and 'id' not in col.lower():
            numerical_cols.remove(col)
            categorical_cols.append(col)

    return numerical_cols, categorical_cols, text_cols, boolean_cols, sparse_cols

# Step 2: Detect Useless Columns
def detect_useless_columns(df, correlation_threshold=0.95, variance_threshold=1e-5, uniqueness_threshold=0.98):
    serial_cols = []
    random_id_cols = []
    low_variance_cols = []
    correlated_cols = set()
    sparse_cols = []

    for col in df.columns:
        if df[col].isna().mean() > 0.9:
            sparse_cols.append(col)

    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        if col in sparse_cols:
            continue
        diffs = df[col].dropna().diff().dropna()
        if len(diffs) > 0 and (diffs.abs() == 1).mean() > 0.9:
            serial_cols.append(col)

    for col in df.columns:
        if col in sparse_cols:
            continue
        unique_ratio = df[col].nunique() / len(df[col].dropna())
        if unique_ratio > uniqueness_threshold:
            random_id_cols.append(col)

    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        if col in sparse_cols:
            continue
        col_std = df[col].std()
        col_mean = df[col].mean()
        if pd.isna(col_std) or (col_std < variance_threshold and col_mean != 0):
            low_variance_cols.append(col)

    numeric_df = df.select_dtypes(include=['int64', 'float64']).drop(columns=sparse_cols)
    if len(numeric_df.columns) > 1 and len(numeric_df) > 10:
        corr_matrix = numeric_df.corr(numeric_only=True).abs()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if corr_matrix.iloc[i, j] > correlation_threshold:
                    correlated_cols.add(corr_matrix.columns[i])

    return serial_cols, random_id_cols, low_variance_cols, list(correlated_cols), sparse_cols

# Step 3: Missing Value Analysis
def analyze_missing_values(df):
    missing_counts = df.isna().sum()
    total_rows = len(df)
    missing_info = []

    for col, missing in missing_counts.items():
        if missing > 0:
            missing_percent = (missing / total_rows) * 100
            category = "Low" if missing_percent <= 10 else "Medium" if missing_percent <= 50 else "High"
            missing_info.append((col, missing, f"{missing_percent:.2f}%", category))

    return missing_info

def analyze_data_distribution(df):
    numerical_df = df.select_dtypes(include=['int64', 'float64'])
    if numerical_df.empty:
        return None

    distribution_stats = []
    for col in numerical_df.columns:
        if numerical_df[col].nunique() > 1:  # Avoid single-value columns
            mean = numerical_df[col].mean()
            std = numerical_df[col].std()
            skewness = stats.skew(numerical_df[col].dropna())
            kurtosis = stats.kurtosis(numerical_df[col].dropna())
            distribution_stats.append((col, mean, std, skewness, kurtosis))

    return pd.DataFrame(distribution_stats, columns=["Column", "Mean", "Std Dev", "Skewness", "Kurtosis"])

def detect_redundant_columns(df, threshold=0.99):
    numerical_df = df.select_dtypes(include=['int64', 'float64'])
    if numerical_df.shape[1] < 2:
        return []

    corr_matrix = numerical_df.corr(numeric_only=True).abs()
    redundant_cols = set()

    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if corr_matrix.iloc[i, j] > threshold:
                redundant_cols.add(corr_matrix.columns[i])

    return list(redundant_cols)

# Step 5: Generate Human-Readable Summary
def generate_data_summary(df, num_cols, cat_cols, text_cols, bool_cols, sparse_cols, redundant_cols):
    summary = "### 📊 AI-Generated Data Analysis Report\n\n"
    summary += f"📂 **Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.**\n\n"

    # Column classifications
    summary += f"🔢 **Numerical Columns:** {len(num_cols)} found.\n"
    summary += f"🔠 **Categorical Columns:** {len(cat_cols)} detected.\n"
    summary += f"📝 **Text-Based Columns:** {len(text_cols)} identified.\n"
    summary += f"✅ **Boolean Columns:** {len(bool_cols)} present.\n"
    summary += f"⚠️ **Sparse Columns (mostly NaN):** {len(sparse_cols)}.\n\n"

    # Preprocessing suggestions
    if len(sparse_cols) > 0:
        summary += f"⚠ Consider **removing sparse columns** {sparse_cols} as they have too many missing values.\n"
    if len(redundant_cols) > 0:
        summary += f"🔗 Columns {redundant_cols} are highly correlated and might be **redundant**.\n"
    if len(num_cols) > 0:
        summary += "📉 **Check numerical columns for outliers** using box plots before analysis.\n"
    if len(text_cols) > 0:
        summary += "🔍 **Review text-based columns** for useful information or consider NLP techniques.\n"

    summary += "\n### 🚀 Suggested Next Steps\n"
    summary += "- Handle missing values appropriately.\n"
    summary += "- Normalize or scale numerical data if needed.\n"
    summary += "- Convert boolean/text values into proper formats for analysis.\n"
    
    return summary


# Streamlit UI
st.title("📊 AI-Powered Data Profiler")

# File uploader
uploaded_file = st.file_uploader("📂 Upload your dataset (CSV, Excel, JSON)", type=["csv", "xlsx", "json"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_json(uploaded_file)

        st.write("### 🔍 Sample Data Preview:")
        st.dataframe(df.head())

        # Step 1: Column Classification
        with st.expander("🔹 Step 1: Column Classification"):
            cat_threshold = st.slider("📏 Categorical Unique Value Ratio Threshold", 0.01, 0.5, 0.1, key="cat_threshold")
            num_cols, cat_cols, text_cols, bool_cols, sparse_cols_step1 = classify_columns(df, cat_threshold)

            st.write("### 📋 Results:")
            st.write(f"📊 **Numerical Columns:** {num_cols}")
            st.write(f"🔠 **Categorical Columns:** {cat_cols}")
            st.write(f"📝 **Text-Based Columns:** {text_cols}")
            st.write(f"✅ **Boolean Columns:** {bool_cols}")
            st.write(f"⚠️ **Sparse Columns (mostly NaN):** {sparse_cols_step1}")

        # Step 2: Useless Column Detection
        with st.expander("🔸 Step 2: Useless Column Detection"):
            corr_threshold = st.slider("📉 Correlation Threshold", 0.5, 1.0, 0.95, key="corr_threshold")
            var_threshold = st.slider("📏 Variance Threshold", 1e-6, 1e-2, 1e-5, key="var_threshold", format="%.6f")
            unique_threshold = st.slider("🆔 Uniqueness Threshold for IDs", 0.9, 1.0, 0.98, key="unique_threshold")

            serial_cols, random_id_cols, low_var_cols, corr_cols, sparse_cols_step2 = detect_useless_columns(
                df, corr_threshold, var_threshold, unique_threshold
            )

            st.write("### 📋 Results:")
            st.write(f"🔢 **Serial Number Columns:** {serial_cols}")
            st.write(f"🆔 **Random ID Columns:** {random_id_cols}")
            st.write(f"📉 **Low-Variance Columns:** {low_var_cols}")
            st.write(f"🔗 **Highly Correlated Columns:** {corr_cols}")
            st.write(f"⚠️ **Sparse Columns (mostly NaN):** {sparse_cols_step2}")

        # Step 3: Missing Value Analysis
        with st.expander("🟠 Step 3: Missing Value Analysis"):
            missing_data = analyze_missing_values(df)
            if missing_data:
                missing_df = pd.DataFrame(missing_data, columns=["Column", "Missing Count", "Missing Percentage", "Impact Level"])
                st.write("### 📋 Missing Value Summary:")
                st.dataframe(missing_df)

                st.write("### 🛠 Suggested Actions:")
                for col, _, percent, impact in missing_data:
                    if impact == "Low":
                        st.write(f"✔ **{col}** ({percent} missing) → Fill with **mean/median/mode**.")
                    elif impact == "Medium":
                        st.write(f"⚠ **{col}** ({percent} missing) → Consider **filling or dropping based on importance**.")
                    else:
                        st.write(f"❌ **{col}** ({percent} missing) → Likely better to **drop the column**.")

            else:
                st.success("✅ No missing values found!")
        
        # Step 4: Data Distribution & Redundancy
        with st.expander("🔹 Step 4: Data Distribution & Redundancy Analysis"):
            st.write("#### 📊 Numerical Data Distribution:")
            dist_stats_df = analyze_data_distribution(df)

            if dist_stats_df is not None:
                st.dataframe(dist_stats_df)

                # Visualization - Histograms
                st.write("### 📉 Histogram of Numerical Columns:")
                for col in dist_stats_df["Column"]:
                    fig, ax = plt.subplots()
                    sns.histplot(df[col].dropna(), kde=True, bins=30, ax=ax)
                    plt.title(f"Distribution of {col}")
                    st.pyplot(fig)

                # Box Plot for Outliers
                st.write("### 📦 Box Plot for Outliers:")
                for col in dist_stats_df["Column"]:
                    fig, ax = plt.subplots()
                    sns.boxplot(x=df[col].dropna(), ax=ax)
                    plt.title(f"Box Plot of {col}")
                    st.pyplot(fig)

            else:
                st.warning("⚠ No numerical data found for analysis.")

            # Redundant Columns Detection
            st.write("#### 🔄 Detecting Redundant Columns")
            redundancy_threshold = st.slider("🔗 Redundancy Correlation Threshold", 0.8, 1.0, 0.99, key="redundancy_threshold")
            redundant_cols = detect_redundant_columns(df, redundancy_threshold)

            if redundant_cols:
                st.warning(f"⚠ **Redundant Columns Detected:** {redundant_cols}")
                st.write("🔹 These columns are highly correlated with others and might be unnecessary.")
            else:
                st.success("✅ No redundant columns found.")

         # Step 5: AI Summary Report
        with st.expander("📝 Step 5: AI-Generated Data Summary"):
            num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            cat_cols = df.select_dtypes(include=['object']).columns.tolist()
            bool_cols = df.select_dtypes(include=['bool']).columns.tolist()
            text_cols = [col for col in cat_cols if df[col].astype(str).str.len().mean() > 50]
            sparse_cols = [col for col in df.columns if df[col].isna().mean() > 0.9]

            summary_text = generate_data_summary(df, num_cols, cat_cols, text_cols, bool_cols, sparse_cols, redundant_cols)
            st.markdown(summary_text)
            
    except Exception as e:
        st.error(f"❌ Error loading file: {e}")
