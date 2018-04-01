import pandas as pd
import numpy as np

df = pd.read_csv("./data/test.csv")

df["is_click"] = 0
df["is_open"] = 0
df = df[["id","is_click"]]

df.to_csv("./data/basic_submit.csv", header=True, index=False)