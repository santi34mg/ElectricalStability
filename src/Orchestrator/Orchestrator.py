import threading
import queue
import os
import pandas as pd
import logging
from typing import List, Optional
from datetime import datetime
from DataLoader.DataLoader import DataLoader
from Processor.Pipeline import Pipeline

SENTINEL = object()
DEFAULT_RETRIES = int(os.getenv("PIPELINE_RETRIES", 3))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class Orchestrator:
    def __init__(self, pipeline: Pipeline, queue_size: int = 8):
        self.pipeline = pipeline
        self.queue = queue.Queue(maxsize=queue_size)
        self.sources: List[threading.Thread] = []
        self.processor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._serialize_func: Optional[callable] = None
        self.batch_prefix = "batch"

    def add_source(self, loader: DataLoader):
        if not isinstance(loader, DataLoader):
            raise TypeError("loader must be a DataLoader subclass")
        t = threading.Thread(target=self._source_worker, args=(loader,), daemon=True)
        self.sources.append(t)
        return self

    def set_serialization(self, func, batch_prefix: str = "batch"):
        self._serialize_func = func
        self.batch_prefix = batch_prefix
        return self

    def _source_worker(self, loader: DataLoader):
        try:
            for df in loader:
                self.queue.put(df)
        except Exception as e:
            logger.exception(f"{loader.__class__.__name__} raised an exception in thread: {e}")
        finally:
            logger.info(f"{loader.__class__.__name__} source finished")

    def _process_loop(self):
        while not self._stop_event.is_set():
            try:
                item = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if item is SENTINEL:
                self._stop_event.set()
                break

            if not isinstance(item, pd.DataFrame):
                self.queue.task_done()
                continue

            retries = 0
            while retries <= DEFAULT_RETRIES:
                try:
                    df = self.pipeline.run(item)
                    if df is not None and not df.empty:
                        self._serialize(df)
                    break
                except Exception as e:
                    retries += 1
                    logger.warning(f"Processing batch failed, attempt {retries}: {e}")
                    if retries > DEFAULT_RETRIES:
                        logger.error(f"Batch failed after {DEFAULT_RETRIES} retries, skipping.")
            self.queue.task_done()

    def _serialize(self, df: pd.DataFrame):
        if self._serialize_func:
            name = f"{self.batch_prefix}_{datetime.utcnow():%Y%m%dT%H%M%S%f}"
            self._serialize_func(df, name)

    def start(self):
        """Starts all source threads and the processor thread."""
        self._stop_event.clear()
        for t in self.sources:
            t.start()
        self.processor_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.processor_thread.start()
        return self

    def stop(self):
        """Stops processing manually."""
        self._stop_event.set()
        self.queue.put(SENTINEL)
        if self.processor_thread:
            self.processor_thread.join(timeout=5)
        for t in self.sources:
            if t.is_alive():
                t.join(timeout=1)
