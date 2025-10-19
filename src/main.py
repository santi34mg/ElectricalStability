import argparse
import logging
import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Example pipeline functions
def normalize(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include="number").columns
    if len(numeric) > 0:
        df[numeric] = (df[numeric] - df[numeric].min()) / (df[numeric].max() - df[numeric].min()).replace(0, 1)
    return df

def profiling(df: pd.DataFrame) -> pd.DataFrame:
    df["_rows"] = len(df)
    return df


def predict_stub(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["prediction"] = 0
    return df


def _ensure_output_dirs(output_dir: Path) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    fig_dir = output_dir / "figures"
    report_dir = output_dir / "reports"
    fig_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir, report_dir


def numeric_stats(df: pd.DataFrame, report_dir: Path) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    stats = df[numeric_cols].describe().T
    stats.to_csv(Path(report_dir) / "numeric_stats.csv")
    return stats


def categorical_stats(df: pd.DataFrame, report_dir: Path) -> dict:
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    cat_summary = {}
    for col in cat_cols:
        counts = df[col].value_counts(dropna=False)
        dfc = pd.DataFrame({"count": counts, "percentage": counts / counts.sum() * 100})
        dfc.to_csv(Path(report_dir) / f"categorical_{col}_summary.csv")
        cat_summary[col] = dfc
    return cat_summary


def correlation(df: pd.DataFrame, fig_dir: Path, report_dir: Path) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numeric_cols].corr()
    corr.to_csv(Path(report_dir) / "correlation.csv")

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title("Correlation matrix")
    plt.tight_layout()
    plt.savefig(Path(fig_dir) / "correlation_heatmap.png")
    plt.close()
    return corr


def missing_values_summary(df: pd.DataFrame, fig_dir: Path, report_dir: Path) -> pd.DataFrame:
    missing = df.isna().sum()
    perc = missing / len(df) * 100
    summary = pd.DataFrame({"missing_count": missing, "missing_percentage": perc})
    summary.to_csv(Path(report_dir) / "missing_values_summary.csv")

    mv = summary[summary["missing_count"] > 0].sort_values("missing_percentage", ascending=False)
    if not mv.empty:
        plt.figure(figsize=(8, max(4, len(mv) * 0.3)))
        sns.barplot(x="missing_percentage", y=mv.index, data=mv, palette="Reds_r")
        plt.xlabel("% missing")
        plt.title("Missing values by column")
        plt.tight_layout()
        plt.savefig(Path(fig_dir) / "missing_values_by_column.png")
        plt.close()

    return summary


def visualize_distributions(df: pd.DataFrame, fig_dir: Path, sample_size: int | None = None, max_columns: int = 12):
    sdf = df if sample_size is None else df.sample(min(sample_size, len(df)))
    numeric_cols = sdf.select_dtypes(include=[np.number]).columns[:max_columns]
    cat_cols = sdf.select_dtypes(include=["object", "category"]).columns[:max_columns]

    for col in numeric_cols:
        plt.figure(figsize=(6, 3))
        sns.histplot(sdf[col].dropna(), kde=True)
        plt.title(f"Histogram of {col}")
        plt.tight_layout()
        safe_name = str(col).replace(os.sep, "_")
        plt.savefig(Path(fig_dir) / f"hist_{safe_name}.png")
        plt.close()

    for col in cat_cols:
        plt.figure(figsize=(6, 3))
        order = sdf[col].value_counts().index
        sns.countplot(y=col, data=sdf, order=order)
        plt.title(f"Bar plot of {col}")
        plt.tight_layout()
        safe_name = str(col).replace(os.sep, "_")
        plt.savefig(Path(fig_dir) / f"count_{safe_name}.png")
        plt.close()


def outliers_iqr(df: pd.DataFrame, col: str, report_dir: Path) -> pd.DataFrame:
    if col not in df.columns:
        return pd.DataFrame()
    if not np.issubdtype(df[col].dtype, np.number):
        return pd.DataFrame()
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    mask = (df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)
    out = df[mask]
    if not out.empty:
        out.to_csv(Path(report_dir) / f"outliers_{col}.csv", index=False)
    return out


def target_analysis(df: pd.DataFrame, target_col: str, report_dir: Path) -> pd.DataFrame | None:
    if target_col not in df.columns:
        return None
    counts = df[target_col].value_counts(dropna=False)
    percentages = counts / counts.sum() * 100
    res = pd.DataFrame({"count": counts, "percentage": percentages})
    res.to_csv(Path(report_dir) / f"target_analysis_{target_col}.csv")
    return res


def detect_candidate_target(df: pd.DataFrame) -> str | None:
    """Try to automatically detect a likely target column name.

    Heuristic: column named 'target' or 'label' or with 2 unique values and not too many unique entries.
    """
    candidates = [c for c in df.columns if c.lower() in ("target", "label", "y", "outcome")]
    if candidates:
        return candidates[0]
    for c in df.columns:
        nunique = df[c].nunique(dropna=False)
        if nunique == 2 and not np.issubdtype(df[c].dtype, np.number):
            return c
    return None


def main(argv: list | None = None):
    parser = argparse.ArgumentParser(description="Run EDA on a CSV dataset")
    parser.add_argument("--data", "-d", help="Path to CSV file", default=None)
    parser.add_argument("--target", "-t", help="Target column name (optional)", default=None)
    parser.add_argument("--sample", "-s", help="Sample size for plotting", type=int, default=1000)
    parser.add_argument("--output", "-o", help="Output directory for reports and figures", default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Resolve default paths
    base = Path(__file__).resolve().parent.parent
    data_path = Path(args.data) if args.data else base / "data" / "Tema_16.csv"
    output_dir = Path(args.output) if args.output else base / "outputs"

    logging.info(f"Loading data from: {data_path}")
    if not data_path.exists():
        logging.error(f"Data file not found: {data_path}")
        return

    df = pd.read_csv(data_path)
    logging.info(f"Loaded dataframe with shape: {df.shape}")

    # quick overview print (head and dtypes)
    logging.info("First 5 rows:\n%s", df.head().to_string())
    logging.info("Data types:\n%s", df.dtypes)

    fig_dir, report_dir = _ensure_output_dirs(output_dir)

    # numerical and categorical stats
    num_stats = numeric_stats(df, report_dir)
    logging.info("Numeric stats saved to %s", report_dir / "numeric_stats.csv")

    cat_stats = categorical_stats(df, report_dir)
    logging.info("Categorical summaries saved to %s", report_dir)

    # missing values
    mv = missing_values_summary(df, fig_dir, report_dir)
    logging.info("Missing values summary saved to %s", report_dir / "missing_values_summary.csv")

    # correlation
    corr = correlation(df, fig_dir, report_dir)
    logging.info("Correlation matrix and heatmap saved")

    # visualizations (use sample size to keep plots fast)
    visualize_distributions(df, fig_dir, sample_size=args.sample)
    logging.info("Distribution and count plots saved to %s", fig_dir)

    # outliers: check numeric columns and save top columns with outliers
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outlier_counts = {}
    for col in numeric_cols:
        out = outliers_iqr(df, col, report_dir)
        outlier_counts[col] = len(out)

    outlier_df = pd.Series(outlier_counts).sort_values(ascending=False)
    outlier_df.to_csv(Path(report_dir) / "outlier_counts_by_column.csv")
    logging.info("Outlier summaries saved to %s", Path(report_dir) / "outlier_counts_by_column.csv")

    # target analysis if available
    target_col = args.target or detect_candidate_target(df)
    if target_col:
        ta = target_analysis(df, target_col, report_dir)
        logging.info("Target analysis for %s saved", target_col)
    else:
        logging.info("No target column detected or provided; skipping target analysis")

    logging.info("EDA complete. Reports and figures are in: %s", output_dir)


if __name__ == "__main__":
    main()