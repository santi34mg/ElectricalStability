from abc import ABC, abstractmethod
from typing import Iterator
import pandas as pd

class DataLoader(ABC):
    """
    Abstract base. Implementations yield pandas DataFrame objects.
    """
    @abstractmethod
    def __iter__(self) -> Iterator[pd.DataFrame]:
        raise NotImplementedError
