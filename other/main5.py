import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import LabelEncoder
import io

# Streamlit Config
st.set_page_config(page_title="Grok 3 Data Profiler", layout="wide")
st.title("🕵️‍♂️ Grok 3’s Ultimate Data Profiler")
st.markdown("Upload your dataset and let me uncover its secrets—no constraints, just insights!")

# Sidebar for Settings
with st.sidebar:
    st.header("⚙️ Profiler Controls")
    plot_type = st.selectbox("Plot Type", ["Histogram", "Box Plot", "Bar Chart (Categorical)"])
    bins = st.slider("Histogram Bins", 10, 50, 20)
    corr_threshold = st.slider("Correlation Threshold", 0.7, 1.0, 0.85)
    missing_strategy = st.selectbox("Missing Value Strategy", ["Median", "Mean", "Mode", "Drop"])

# File Uploader
uploaded_file = st.file_uploader("📂 Upload Dataset (CSV, Excel, JSON)", type=["csv", "xlsx", "json"])

if uploaded_file:
    try:
        # Load Data
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_json(uploaded_file)
        
        st.success(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns!")

        # Tabs for Navigation
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "Raw Peek", "Blueprint", "Health Check", "Stats & Insights", "Relationships", "Summary"
        ])

        # Step 1: Raw Peek
        with tab1:
            st.subheader("1. First Encounter – The Raw Peek")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("**Head (First 5)**")
                st.dataframe(df.head())
            with col2:
                st.write("**Tail (Last 5)**")
                st.dataframe(df.tail())
            with col3:
                st.write("**Random Sample (5)**")
                st.dataframe(df.sample(5))
            # Chaos Index (simple version)
            chaos = df.apply(lambda x: x.apply(lambda y: isinstance(y, str) and x.dtype != 'object').sum()).sum()
            st.info(f"Chaos Index: {chaos} cells deviate from expected types.")

        # Step 2: Structural Blueprint
        with tab2:
            st.subheader("2. Structural Blueprint – Mapping the Terrain")
            num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            cat_cols = df.select_dtypes(include=['object']).columns.tolist()
            date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
            sparse_cols = [col for col in df.columns if df[col].isna().mean() > 0.8]
            
            st.write(f"📏 **Size**: {df.shape[0]} rows, {df.shape[1]} columns")
            st.write(f"📊 **Numerical**: {len(num_cols)} - {', '.join(num_cols[:3]) + '...' if num_cols else 'None'}")
            st.write(f"🔠 **Categorical**: {len(cat_cols)} - {', '.join(cat_cols[:3]) + '...' if cat_cols else 'None'}")
            st.write(f"⏳ **Temporal**: {len(date_cols)} - {', '.join(date_cols[:3]) + '...' if date_cols else 'None'}")
            st.write(f"⚠ **Sparse**: {len(sparse_cols)} - {', '.join(sparse_cols[:3]) + '...' if sparse_cols else 'None'}")

        # Step 3: Health Check
        with tab3:
            st.subheader("3. Vital Signs – Health Check")
            missing = df.isna().sum()
            duplicates = df.duplicated().sum()
            trust_scores = {col: 100 * (1 - missing[col]/len(df) - df[col].nunique()/len(df) if df[col].dtype == 'object' else 0) 
                           for col in df.columns}
            
            st.write("**Missing Values**")
            st.dataframe(pd.DataFrame([(col, count, f"{count/len(df)*100:.1f}%") 
                                       for col, count in missing.items() if count > 0], 
                                     columns=["Column", "Missing", "%"]))
            st.write(f"**Duplicates**: {duplicates} rows")
            st.write("**Trust Scores** (0-100, based on completeness & variety)")
            st.dataframe(pd.DataFrame(trust_scores.items(), columns=["Column", "Trust Score"]))

            # Visual: Missingness Heatmap
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(df.isna(), cbar=False, cmap="binary", ax=ax)
            st.pyplot(fig)

        # Step 4: Stats & Insights
        with tab4:
            st.subheader("4. Pulse Check – Descriptive Heartbeat")
            if num_cols:
                stats = df[num_cols].agg(['mean', 'std', 'min', 'max', skew, kurtosis]).T
                stats.columns = ['Mean', 'Std', 'Min', 'Max', 'Skew', 'Kurtosis']
                st.write("**Numerical Stats**")
                st.dataframe(stats)

                # Plotting
                col_to_plot = st.selectbox("Select Column to Plot", num_cols + cat_cols)
                if col_to_plot in num_cols and plot_type == "Histogram":
                    fig, ax = plt.subplots()
                    sns.histplot(df[col_to_plot].dropna(), bins=bins, ax=ax)
                    st.pyplot(fig)
                elif col_to_plot in num_cols and plot_type == "Box Plot":
                    fig, ax = plt.subplots()
                    sns.boxplot(x=df[col_to_plot].dropna(), ax=ax)
                    st.pyplot(fig)
                elif col_to_plot in cat_cols and plot_type == "Bar Chart (Categorical)":
                    fig, ax = plt.subplots()
                    df[col_to_plot].value_counts().plot.bar(ax=ax)
                    st.pyplot(fig)

            if cat_cols:
                st.write("**Categorical Insights**")
                for col in cat_cols[:3]:
                    st.write(f"{col}: Top 5 - {df[col].value_counts().head().to_dict()}")

        # Step 5: Relationships
        with tab5:
            st.subheader("5. Connections – The Web of Relationships")
            if len(num_cols) > 1:
                corr_matrix = df[num_cols].corr()
                st.write("**Correlation Matrix**")
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
                st.pyplot(fig)
                high_corr = [(c1, c2, corr_matrix.loc[c1, c2]) 
                            for c1 in num_cols for c2 in num_cols 
                            if c1 < c2 and abs(corr_matrix.loc[c1, c2]) > corr_threshold]
                if high_corr:
                    st.warning(f"High Correlations: {high_corr}")

        # Step 6: Summary & Narrative
        with tab6:
            st.subheader("6. Living Report – The Narrative")
            health_score = 100 * (1 - len([m for m in missing if m > 0])/df.shape[1] - duplicates/len(df))
            narrative = f"""
            ### 📊 Data Profile Report – March 24, 2025
            **What It Is**: A dataset with {df.shape[0]} rows and {df.shape[1]} columns.  
            **Health Score**: {health_score:.0f}/100 – penalized for missing data and duplicates.  
            **Standouts**: {len(num_cols)} numerical columns, {len(cat_cols)} categorical, {len(sparse_cols)} sparse.  
            **Issues**: {sum(m > 0 for m in missing.values())} columns with missing values, {duplicates} duplicates.  
            **Next Steps**: 
            - Clean sparse columns: {sparse_cols[:3] if sparse_cols else 'None'}.
            - Handle missing with {missing_strategy.lower()}.
            - Investigate high correlations: {high_corr[:2] if high_corr else 'None'}.
            """
            st.markdown(narrative)

            # Scenario: Handle Missing Values
            df_clean = df.copy()
            for col in num_cols:
                if missing_strategy == "Median":
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                elif missing_strategy == "Mean":
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
                elif missing_strategy == "Drop":
                    df_clean = df_clean.dropna(subset=[col])
            for col in cat_cols:
                if missing_strategy == "Mode":
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

            # Download Options
            st.download_button("📥 Download Report", narrative, "data_profile.txt")
            csv = df_clean.to_csv(index=False)
            st.download_button("📤 Download Cleaned Data", csv, "cleaned_data.csv")

    except Exception as e:
        st.error(f"❌ Error: {e}")

else:
    st.info("Upload a dataset to begin profiling!")