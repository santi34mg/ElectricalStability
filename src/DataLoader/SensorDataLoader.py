import pandas as pd
import time
from typing import Iterator
from .DataLoader import DataLoader

class SensorDataLoader(DataLoader):
    """
    Accepts a generator/callback that yields dict-like sensor readings.
    Each reading is converted to a single-row DataFrame and yielded.
    """
    def __init__(self, reading_generator, interval: float = 0.0):
        self.gen = reading_generator
        self.interval = interval

    def __iter__(self) -> Iterator[pd.DataFrame]:
        for reading in self.gen():
            df = pd.DataFrame([reading])
            yield df
            if self.interval:
                time.sleep(self.interval)