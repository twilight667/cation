import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks

# Load data
df = pd.read_excel("arya_ecg_data.xlsx")

# Rename columns properly
df.columns = ["timestamp", "adc"]

# Convert to numeric and drop errors
df["adc"] = pd.to_numeric(df["adc"], errors="coerce")
df = df.dropna()

signal = df["adc"].values

# ---- STEP B: Plot raw signal ----
plt.figure(figsize=(12, 4))
plt.plot(signal, linewidth=0.8)
plt.title("Raw ECG Signal (Unfiltered)")
plt.xlabel("Samples")
plt.ylabel("ADC Value")
plt.grid(True)
plt.tight_layout()
plt.show()
