# 📊 EDA Automation Tool (Streamlit)

An interactive **Streamlit web application** that automates **Exploratory Data Analysis (EDA)** for any uploaded CSV file.  
It provides dataset diagnostics, missing value analysis, statistical summaries, and visual insights through univariate and bivariate analysis.

---

## 🚀 Features

### ✅ Dataset Overview
- Dataset shape (rows, columns)
- Data preview (top rows)
- Schema summary:
  - Column names
  - Data types
  - Missing value counts and percentages
- Smart warnings:
  - Columns with high missing values
  - High-cardinality categorical columns (ID-like features)

---

### ✅ Univariate Analysis
#### Numeric Columns
- Summary statistics: count, mean, median, std, skewness
- Histograms with KDE
- Boxplots for outlier visualization
- Skewness interpretation labels

#### Categorical Columns
- Unique count and mode
- Top-N category distributions
- Percent share + “Others” bucket
- High-cardinality warnings

---

### ✅ Correlation Analysis
- Pearson and Spearman correlation options
- Masked correlation heatmap (lower triangle)
- Top correlated feature pairs table (interactive Top-N)

---

### ✅ Bivariate Analysis
#### Numeric vs Numeric
- Scatter plot
- Correlation value

#### Categorical vs Numeric
- Grouped boxplot (numeric distribution across categories)
- Option to limit category count for readability

#### Categorical vs Categorical
- Crosstab table
- Stacked bar chart (count or percentage mode)

---

## 🧠 Tech Stack

- **Python**
- **Streamlit**
- **Pandas / NumPy**
- **Matplotlib**
- **Seaborn**

---

## 📂 Project Structure

```

eda-automation-tool/
│
├── main.py
├── requirements.txt
└── README.md

````

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the app

```bash
streamlit run main.py
```

---

## 📌 Use Cases

* Quick dataset understanding before ML modeling
* Missing value diagnostics & schema validation
* Feature relationship analysis
* Detecting outliers and class/category imbalance


## 👤 Author

**Ishatva Singh Panwar**

