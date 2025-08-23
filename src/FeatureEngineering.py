#python FeatureEngineering.py --csv "../data/datafile.csv"
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.model_selection import KFold

DATA_DIR = "../data/datafile"  
PROC_DIR = os.path.join("data", "processed")
FIG_DIR = os.path.join("reports", "figures")

os.makedirs(PROC_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


def load_data(csv_path: str = None) -> pd.DataFrame:
    """Load dataset from CSV inside given folder"""
    if csv_path is None:
        candidates = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".csv")]
        if not candidates:
            raise FileNotFoundError(f"No CSV found in {DATA_DIR}.")
        csv_path = os.path.join(DATA_DIR, candidates[0])
    df = pd.read_csv(csv_path)
    return df


def basic_inspect(df: pd.DataFrame):
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print("\nHead:\n", df.head())
    print("\nInfo:\n")
    df.info()
    print("\nDescribe:\n", df.describe(include="all"))

    



def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names and fix anomalies"""
    df = df.copy()
    df.columns = [c.strip().replace("-", "_").replace(" ", "_").lower() for c in df.columns]

    # Fix datetime
    for dt_col in ["ScheduledDay","AppointmentDay"]:
        if dt_col in df.columns:
            df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")

    # Fix Age
    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
        df.loc[(df["age"] < 0) | (df["age"] > 100), "age"] = np.nan

    # Fix target column
    if "no_show" in df.columns:
        df["no_show"] = df["no_show"].astype(str).str.upper().map({"NO": 0, "YES": 1})

    # Drop duplicates
    df = df.drop_duplicates()
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering"""
    df = df.copy()

    if "ScheduledDay" in df.columns and "AppointmentDay" in df.columns:
        df["waiting_days"] = (df["AppointmentDay"].dt.normalize() - df["ScheduledDay"].dt.normalize()).dt.days
        df["waiting_hours"] = (df["AppointmentDay"] - df["ScheduledDay"]).dt.total_seconds() / 3600.0
        df["same_day"] = (df["waiting_days"] == 0).astype(int)

    if "age" in df.columns:
        bins = [0, 5, 12, 18, 30, 45, 60, 75, 100]
        labels = ["0-5", "6-12", "13-18", "19-30", "31-45", "46-60", "61-75", "76-100"]
        df["age_bucket"] = pd.cut(df["age"], bins=bins, labels=labels, include_lowest=True)

    return df


def save_processed(df_clean: pd.DataFrame, df_feat: pd.DataFrame):
    clean_path = os.path.join(PROC_DIR, "cleaned_data.csv")
    feat_path = os.path.join(PROC_DIR, "features_data.csv")
    df_clean.to_csv(clean_path, index=False)
    df_feat.to_csv(feat_path, index=False)
    print(f"Saved cleaned → {clean_path}")
    print(f"Saved features → {feat_path}")


def matplotlib_plots(df: pd.DataFrame):
    """Basic Matplotlib plots"""
    num_cols = df.select_dtypes(include=[np.number]).columns

    # Histograms
    for col in num_cols[:5]:
        plt.figure()
        df[col].dropna().hist(bins=30)
        plt.title(f"Histogram — {col}")
        plt.savefig(os.path.join(FIG_DIR, f"mpl_hist_{col}.png"))
        plt.close()

    # Correlation heatmap
    if len(num_cols) > 1:
        corr = df[num_cols].corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=False, cmap="coolwarm")
        plt.title("Correlation Heatmap")
        plt.savefig(os.path.join(FIG_DIR, "mpl_corr_heatmap.png"))
        plt.close()


def plotly_plots(df: pd.DataFrame):
    """Interactive Plotly plots"""
    if "age" in df.columns and "waiting_days" in df.columns:
        fig = px.scatter(df, x="age", y="waiting_days", color="no_show", title="Age vs Waiting Days")
        fig.write_html(os.path.join(FIG_DIR, "plotly_scatter_age_waiting.html"))


def run_pipeline(csv_path: str = None):
    df = load_data(csv_path)
    basic_inspect(df)

    df_clean = clean_data(df)
    df_feat = engineer_features(df_clean)

    save_processed(df_clean, df_feat)

    print("Generating Matplotlib plots...")
    matplotlib_plots(df_feat)

    print("Generating Plotly plots...")
    plotly_plots(df_feat)

    print(f"All figures saved to {FIG_DIR}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None, help="Path to CSV (optional)")
    args = parser.parse_args()
    run_pipeline(args.csv)


if __name__ == "__main__":
    main()
