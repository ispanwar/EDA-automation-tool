import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.set_page_config(page_title="EDA Automation Tool", page_icon="📊", layout="wide")
sns.set_style("whitegrid")
def skewness_label(sk: float):
    if sk >= 1:
        return "Highly right skewed"
    elif 0.5 < sk < 1:
        return "Right skewed"
    elif -0.5 <= sk <= 0.5:
        return "Approx. symmetric"
    elif -1 < sk < -0.5:
        return "Left skewed"
    else:
        return "Highly left skewed"


@st.cache_data(show_spinner=False)
def load_data(file) -> pd.DataFrame:
    return pd.read_csv(file)


@st.cache_data(show_spinner=False)
def build_schema_table(df: pd.DataFrame) -> pd.DataFrame:
    col_names = df.columns.tolist()

    nulls = [int(df[col].isnull().sum()) for col in col_names]
    non_nulls = [int(df.shape[0] - df[col].isnull().sum()) for col in col_names]
    dtypes = [str(df[col].dtype) for col in col_names]
    missing_pct = [(df[col].isnull().sum() / df.shape[0]) * 100 for col in col_names]

    info = {
        "Column Name": col_names,
        "Data Type": dtypes,
        "Non-Null Count": non_nulls,
        "Null Count": nulls,
        "Missing %": np.round(missing_pct, 2),
    }
    schema = pd.DataFrame(info).sort_values(by="Missing %", ascending=False)
    return schema


@st.cache_data(show_spinner=False)
def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    num_df = df.select_dtypes(include=["number"])
    if num_df.shape[1] == 0:
        return pd.DataFrame()

    summary = pd.DataFrame({
        "Count": num_df.notnull().sum(),
        "Mean": num_df.mean(),
        "Median": num_df.median(),
        "Std": num_df.std(),
        "Min": num_df.min(),
        "Max": num_df.max(),
        "Skew": num_df.skew()
    })

    return summary.round(3)


@st.cache_data(show_spinner=False)
def categorical_summary(df: pd.DataFrame) -> pd.DataFrame:
    cat_df = df.select_dtypes(include=["object", "category", "bool"])
    if cat_df.shape[1] == 0:
        return pd.DataFrame()

    out = []
    for col in cat_df.columns:
        series = cat_df[col]
        mode_val = None
        if series.dropna().shape[0] > 0:
            mode_val = series.mode(dropna=True)[0]

        out.append({
            "Column": col,
            "Non-Null Count": int(series.notnull().sum()),
            "Unique Count": int(series.nunique(dropna=True)),
            "Mode": mode_val
        })

    return pd.DataFrame(out).set_index("Column")


def top_category_table(df: pd.DataFrame, col: str, top_n: int = 5) -> pd.DataFrame:
    vc = df[col].value_counts(dropna=True)
    total = int(vc.sum())
    unique_count = int(vc.shape[0])

    if unique_count == 0:
        return pd.DataFrame()

    top_vc = vc.head(top_n)
    top_df = top_vc.reset_index()
    top_df.columns = ["Category", "Count"]
    top_df["%"] = (top_df["Count"] / total) * 100
    top_df["%"] = top_df["%"].round(2)

    if unique_count > top_n:
        others_count = int(total - top_vc.sum())
        others_pct = round((others_count / total) * 100, 2)
        others_row = pd.DataFrame({"Category": ["Others"], "Count": [others_count], "%": [others_pct]})
        top_df = pd.concat([top_df, others_row], ignore_index=True)

    return top_df


def correlation_pairs_table(corr_matrix: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    corr_pairs = corr_matrix.abs().unstack().reset_index()
    corr_pairs.columns = ["Feature 1", "Feature 2", "Correlation"]
    corr_pairs = corr_pairs[corr_pairs["Feature 1"] != corr_pairs["Feature 2"]]
    corr_pairs["Pair"] = corr_pairs.apply(lambda x: "-".join(sorted([x["Feature 1"], x["Feature 2"]])), axis=1)
    corr_pairs = corr_pairs.drop_duplicates(subset="Pair").drop(columns="Pair")
    corr_pairs = corr_pairs.sort_values(by="Correlation", ascending=False).head(top_n)
    corr_pairs["Correlation"] = corr_pairs["Correlation"].round(3)
    return corr_pairs


# UI Components

def overview_tab(df: pd.DataFrame):
    st.subheader("📌 Dataset Overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", f"{df.shape[1]:,}")
    c3.metric("Total Missing Cells", f"{int(df.isnull().sum().sum()):,}")

    st.markdown("### Preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("### Dataset Summary (Schema)")
    schema = build_schema_table(df)
    st.dataframe(schema, use_container_width=True)

    # Quick warnings
    st.markdown("### 🚩 Quick Warnings")
    colA, colB = st.columns(2)

    with colA:
        worst_missing = schema[schema["Missing %"] > 0].head(5)
        if worst_missing.shape[0] == 0:
            st.success("No missing values detected ✅")
        else:
            st.warning("Top Missing Columns")
            st.dataframe(worst_missing[["Column Name", "Missing %", "Null Count"]], use_container_width=True)

    with colB:
        cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        if len(cat_cols) == 0:
            st.info("No categorical columns found.")
        else:
            warnings = []
            total_rows = df.shape[0]
            for col in cat_cols:
                unique_count = int(df[col].nunique(dropna=True))
                unique_ratio = unique_count / total_rows if total_rows else 0
                if unique_count > 50 or unique_ratio > 0.2:
                    warnings.append((col, unique_count, round(unique_ratio * 100, 2)))

            if len(warnings) == 0:
                st.success("No high-cardinality columns ✅")
            else:
                st.warning("High Cardinality Columns")
                warn_df = pd.DataFrame(warnings, columns=["Column", "Unique Values", "Unique % of Rows"])
                st.dataframe(warn_df, use_container_width=True)


def univariate_tab(df: pd.DataFrame):
    st.subheader("📈 Univariate Analysis")

    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    tab1, tab2 = st.tabs(["Numeric", "Categorical"])

    # Numeric Univariate
    with tab1:
        if len(num_cols) == 0:
            st.info("No numeric columns found.")
        else:
            selected = st.multiselect("Select numeric columns", num_cols, default=num_cols[: min(3, len(num_cols))])
            bins = st.slider("Histogram bins", 10, 100, 30)
            show_box = st.checkbox("Show boxplots", value=True)

            for col in selected:
                st.markdown(f"### 🔹 {col}")
                series = df[col].dropna()
                if series.shape[0] < 2:
                    st.warning("Not enough non-null values.")
                    continue

                sk = series.skew()
                sk_label = skewness_label(sk)

                a, b, c, d, e, f = st.columns(6)
                a.metric("Count", f"{series.shape[0]:,}")
                b.metric("Mean", f"{series.mean():.3f}")
                c.metric("Median", f"{series.median():.3f}")
                d.metric("Std", f"{series.std():.3f}")
                e.metric("Skew", f"{sk:.3f}")
                f.metric("Shape", sk_label)

                p1, p2 = st.columns(2)
                with p1:
                    fig, ax = plt.subplots()
                    sns.histplot(series, kde=True, ax=ax)
                    ax.set_title(f"Histogram: {col}")
                    st.pyplot(fig)

                if show_box:
                    with p2:
                        fig, ax = plt.subplots()
                        ax.boxplot(series, vert=True)
                        ax.set_title(f"Boxplot: {col}")
                        st.pyplot(fig)

    # Categorical Univariate
    with tab2:
        if len(cat_cols) == 0:
            st.info("No categorical columns found.")
        else:
            selected = st.multiselect("Select categorical columns", cat_cols, default=cat_cols[: min(3, len(cat_cols))])
            top_n = st.slider("Top N categories", 3, 30, 5)

            for col in selected:
                with st.expander(f"🔎 {col}", expanded=False):
                    unique_count = int(df[col].nunique(dropna=True))
                    unique_ratio = unique_count / df.shape[0] if df.shape[0] else 0

                    st.write(f"**Unique categories:** {unique_count}")
                    if unique_count > 50 or unique_ratio > 0.2:
                        st.warning(f"⚠️ High cardinality column ({unique_count} uniques | {unique_ratio*100:.2f}% of rows).")

                    table = top_category_table(df, col, top_n=top_n)
                    if table.empty:
                        st.warning("No valid values found.")
                    else:
                        st.dataframe(table, use_container_width=True)


def correlation_tab(df: pd.DataFrame):
    st.subheader("🔥 Correlation Analysis")

    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.shape[1] < 2:
        st.info("Need at least 2 numeric columns to compute correlations.")
        return

    corr_method = st.selectbox("Correlation method", ["pearson", "spearman"])
    corr_matrix = numeric_df.corr(method=corr_method)

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    annot_flag = corr_matrix.shape[1] <= 30

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=annot_flag,
        fmt=".2f",
        linewidths=0.5,
        cbar=True,
        ax=ax,
        cmap='mako'
    )
    ax.set_title(f"{corr_method.title()} Correlation Heatmap")
    if not annot_flag:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    st.pyplot(fig)

    st.markdown("### 🔝 Top Correlated Feature Pairs")
    top_n = st.slider("Top N correlations", 5, 30, 10)
    st.dataframe(correlation_pairs_table(corr_matrix, top_n=top_n), use_container_width=True)


def bivariate_tab(df: pd.DataFrame):
    st.subheader("🔁 Bivariate Analysis")

    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    t1, t2, t3 = st.tabs(["Numeric vs Numeric", "Categorical vs Numeric", "Categorical vs Categorical"])

    # Numeric vs Numeric
    with t1:
        if len(num_cols) < 2:
            st.info("Need at least 2 numeric columns.")
        else:
            x_col = st.selectbox("X (numeric)", num_cols, key="x_num")
            y_col = st.selectbox("Y (numeric)", num_cols, index=min(1, len(num_cols)-1), key="y_num")

            if x_col == y_col:
                st.warning("Choose two different numeric columns.")
            else:
                valid = df[[x_col, y_col]].dropna()
                if valid.shape[0] < 2:
                    st.warning("Not enough non-null pairs.")
                else:
                    corr_val = valid[x_col].corr(valid[y_col])
                    st.write(f"**Correlation:** {corr_val:.3f}")

                    fig, ax = plt.subplots()
                    ax.scatter(valid[x_col], valid[y_col])
                    ax.set_xlabel(x_col)
                    ax.set_ylabel(y_col)
                    ax.set_title(f"{x_col} vs {y_col}")
                    st.pyplot(fig)

    # Categorical vs Numeric
    with t2:
        if len(cat_cols) == 0 or len(num_cols) == 0:
            st.info("Need at least one categorical and one numeric column.")
        else:
            cat = st.selectbox("Categorical", cat_cols, key="cat_col")
            num = st.selectbox("Numeric", num_cols, key="num_col")

            temp = df[[cat, num]].dropna()
            if temp.shape[0] < 2:
                st.warning("Not enough data.")
            else:
                max_cats = st.slider("Max categories to plot", 1, 30, 10)
                top_categories = temp[cat].value_counts().head(max_cats).index
                temp = temp[temp[cat].isin(top_categories)]

                fig, ax = plt.subplots(figsize=(12, 5))
                sns.boxplot(data=temp, x=cat, y=num, ax=ax)
                ax.set_title(f"{num} across {cat}")
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
                st.pyplot(fig)

    # Categorical vs Categorical
    with t3:
        if len(cat_cols) < 2:
            st.info("Need at least 2 categorical columns.")
        else:
            c1 = st.selectbox("Categorical 1", cat_cols, key="c1")
            c2 = st.selectbox("Categorical 2", cat_cols, index=min(1, len(cat_cols)-1), key="c2")

            if c1 == c2:
                st.warning("Choose two different categorical columns.")
            else:
                temp = df[[c1, c2]].dropna()
                if temp.shape[0] == 0:
                    st.warning("No valid rows.")
                else:
                    ct = pd.crosstab(temp[c1], temp[c2])
                    st.dataframe(ct, use_container_width=True)
                st.markdown("### 📊 Stacked Bar Chart")

                # Optional: Normalize for percentage view
                show_percent = st.checkbox("Show as percentage", value=False)

                plot_df = ct.copy()

                if show_percent:
                    plot_df = plot_df.div(plot_df.sum(axis=1), axis=0) * 100  # row-wise %

                fig, ax = plt.subplots(figsize=(12, 6))
                plot_df.plot(kind="bar", stacked=True, ax=ax)

                ax.set_title(f"Stacked Bar Chart: {c1} vs {c2}")
                ax.set_xlabel(c1)
                ax.set_ylabel("Percentage (%)" if show_percent else "Count")

                ax.legend(title=c2, bbox_to_anchor=(1.02, 1), loc="upper left")
                plt.xticks(rotation=45, ha="right")

                st.pyplot(fig)


# -----------------------------
# Main App
# -----------------------------
st.title("📊 EDA Automation Tool")
st.caption("Upload a CSV file to automatically explore and analyze your dataset.")

file = st.file_uploader("Upload a CSV file", type=["csv"])

if file is None:
    st.info("Please upload a CSV file to start.")
    st.stop()

try:
    df = load_data(file)

    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Controls")
        show_preview = st.checkbox("Show Data Preview", value=True)
        if show_preview:
            st.write("Preview is shown in Overview tab (top 10 rows).")

    # Tabs
    tab_overview, tab_uni, tab_corr, tab_bi = st.tabs(["📌 Overview", "📈 Univariate", "🔥 Correlation", "🔁 Bivariate"])

    with tab_overview:
        overview_tab(df)

    with tab_uni:
        st.markdown("### Numeric Summary Table")
        ns = numeric_summary(df)
        if ns.empty:
            st.info("No numeric columns found.")
        else:
            st.dataframe(ns, use_container_width=True)

        st.markdown("### Categorical Summary Table")
        cs = categorical_summary(df)
        if cs.empty:
            st.info("No categorical columns found.")
        else:
            st.dataframe(cs, use_container_width=True)

        st.divider()
        univariate_tab(df)

    with tab_corr:
        correlation_tab(df)

    with tab_bi:
        bivariate_tab(df)

except Exception as e:
    st.error(f"Error occurred: {e}")
