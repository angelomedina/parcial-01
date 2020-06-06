import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("dataset.csv")

x= dataset.iloc[:, [0,1] ].values

print(x)