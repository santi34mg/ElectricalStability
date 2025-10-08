import pandas as pd
from typing import Iterator, Optional
from .DataLoader import DataLoader

class CSVDataLoader(DataLoader):
    """
    Iterates CSV in chunks as pandas DataFrame. Does not keep prior chunks.
    """
    def __init__(self, path: str, chunksize: int = 10000, usecols=None, dtype=None):
        self.path = path
        self.chunksize = chunksize
        self.usecols = usecols
        self.dtype = dtype

    def __iter__(self) -> Iterator[pd.DataFrame]:
        for chunk in pd.read_csv(self.path, chunksize=self.chunksize, usecols=self.usecols, dtype=self.dtype):
            yield chunk
