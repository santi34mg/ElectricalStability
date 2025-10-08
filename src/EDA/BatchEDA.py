import pandas as pd
import numpy as np
from collections import defaultdict

class BatchEDA:
    """
    Memory-efficient exploratory data analysis for large datasets.
    Updates statistics incrementally per DataFrame batch.
    """
    def __init__(self):
        self.n_rows = 0
        self.n_cols = None
        self.dtypes = {}
        self.numeric_stats = defaultdict(lambda: {"sum": 0, "sum_sq": 0, "min": np.inf, "max": -np.inf})
        self.categorical_counts = defaultdict(lambda: defaultdict(int))
        self.missing_counts = defaultdict(int)
        self.duplicates = 0
        self.seen_rows = set()

    def update(self, df: pd.DataFrame):
        # Update shape info
        if self.n_cols is None:
            self.n_cols = df.shape[1]
            self.dtypes = df.dtypes.to_dict()
        self.n_rows += len(df)

        # Missing values
        for col in df.columns:
            self.missing_counts[col] += df[col].isna().sum()

        # Numeric stats
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            col_data = df[col].dropna()
            self.numeric_stats[col]["sum"] += col_data.sum()
            self.numeric_stats[col]["sum_sq"] += (col_data ** 2).sum()
            self.numeric_stats[col]["min"] = min(self.numeric_stats[col]["min"], col_data.min())
            self.numeric_stats[col]["max"] = max(self.numeric_stats[col]["max"], col_data.max())

        # Categorical counts
        cat_cols = df.select_dtypes(include=['category']).columns
        for col in cat_cols:
            counts = df[col].value_counts(dropna=False).to_dict()
            for val, cnt in counts.items():
                self.categorical_counts[col][val] += cnt

        # Duplicate rows (simple hash-based detection)
        for row in df.itertuples(index=False, name=None):
            if row in self.seen_rows:
                self.duplicates += 1
            else:
                self.seen_rows.add(row)

    def summarize(self):
        summary = {}

        # Numeric summary
        numeric_summary = {}
        for col, stats in self.numeric_stats.items():
            count = self.n_rows - self.missing_counts.get(col, 0)
            mean = stats["sum"] / count if count > 0 else np.nan
            variance = (stats["sum_sq"] / count - mean**2) if count > 1 else np.nan
            std = np.sqrt(variance) if variance >= 0 else np.nan
            numeric_summary[col] = {
                "mean": mean,
                "std": std,
                "min": stats["min"],
                "max": stats["max"],
                "missing": self.missing_counts.get(col, 0)
            }

        # Categorical summary
        categorical_summary = {}
        for col, counts in self.categorical_counts.items():
            total = sum(counts.values())
            categorical_summary[col] = {"counts": dict(counts), "total": total, "missing": self.missing_counts.get(col, 0)}

        summary["n_rows"] = self.n_rows
        summary["n_cols"] = self.n_cols
        summary["dtypes"] = self.dtypes
        summary["duplicates"] = self.duplicates
        summary["numeric"] = numeric_summary
        summary["categorical"] = categorical_summary
        return summary
