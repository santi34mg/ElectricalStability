import os
import queue
import threading
import pandas as pd
from typing import Optional, Callable
from datetime import datetime
from Processor.Pipeline import Pipeline

SENTINEL = object()

class Processor:
    """
    Pulls DataFrames from a queue, runs them through a Pipeline, and serializes results.
    Thread-safe. No persistent storage in memory.
    """
    def __init__(
        self,
        work_queue: "queue.Queue",
        pipeline: Pipeline,
        output_dir: str = "out",
        serializer: Optional[Callable[[pd.DataFrame, str], None]] = None,
        prefix: str = "batch"
    ):
        self.q = work_queue
        self.pipeline = pipeline
        self.output_dir = output_dir
        self.serializer = serializer or self._to_parquet
        self.prefix = prefix
        self._stop_event = threading.Event()
        os.makedirs(self.output_dir, exist_ok=True)

    def _to_parquet(self, df: pd.DataFrame, name: str):
        path = os.path.join(self.output_dir, f"{name}.parquet")
        df.to_parquet(path, index=False)

    def process_one(self, df: pd.DataFrame):
        df = self.pipeline.run(df)
        if df is None or df.empty:
            return
        name = f"{self.prefix}_{datetime.utcnow():%Y%m%dT%H%M%S%f}"
        self.serializer(df, name)

    def run_forever(self):
        while not self._stop_event.is_set():
            item = self.q.get()
            try:
                if item is SENTINEL:
                    self._stop_event.set()
                    break
                if isinstance(item, pd.DataFrame):
                    self.process_one(item)
            finally:
                self.q.task_done()

    def start_threaded(self, daemon: bool = True):
        t = threading.Thread(target=self.run_forever, daemon=daemon)
        t.start()
        return t

    def stop(self):
        self._stop_event.set()
        self.q.put(SENTINEL)
