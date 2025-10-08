import pandas as pd
from typing import Callable, List

class Pipeline:
    def __init__(self):
        self.steps: List[Callable[[pd.DataFrame], pd.DataFrame]] = []

    def add(self, func: Callable[[pd.DataFrame], pd.DataFrame]):
        self.steps.append(func)
        return self

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        for step in self.steps:
            df = step(df)
            if df is None or df.empty:
                break
        return df
