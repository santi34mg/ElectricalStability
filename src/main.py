import time
import pandas as pd
from DataLoader.CSVDataLoader import CSVDataLoader
from DataLoader.SensorDataLoader import SensorDataLoader
from EDA.BatchEDA import BatchEDA
from Orchestrator.Orchestrator import Orchestrator
from Processor.Pipeline import Pipeline
from pprint import pprint

# Example pipeline functions
def normalize(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include="number").columns
    if numeric.any():
        df[numeric] = (df[numeric] - df[numeric].min()) / (df[numeric].max() - df[numeric].min()).replace(0, 1)
    return df

def profiling(df: pd.DataFrame) -> pd.DataFrame:
    df["_rows"] = len(df)
    return df

def predict_stub(df: pd.DataFrame) -> pd.DataFrame:
    df["prediction"] = 0
    return df

# Example sensor generator
def sensor_generator():
    for i in range(10):
        yield {"sensor_id": "s1", "timestamp": pd.Timestamp.utcnow(), "value": float(i)}
        time.sleep(0.1)

# Main
def main():
    csv_loader = CSVDataLoader("data/Tema_16.csv", chunksize=5000)
    sensor_loader = SensorDataLoader(sensor_generator)

    eda = BatchEDA()
    for batch in csv_loader:
        eda.update(batch)
    
    summary = eda.summarize()
    pprint(summary)

    # pipeline = Pipeline().add(normalize).add(profiling).add(predict_stub)

    # orch = Orchestrator(pipeline)
    # orch.add_source(csv_loader).add_source(sensor_loader)
    # orch.start()

    # # later
    # time.sleep(15)
    # orch.stop()

if __name__ == "__main__":
    main()
