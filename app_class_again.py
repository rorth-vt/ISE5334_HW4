import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft
import pywt

st.set_page_config(page_title="Time-Series Feature Engineering", layout="wide")

st.title("📊 Time-Series Feature Engineering Dashboard")

# =========================
# Data Input
# =========================
st.sidebar.header("Input Options")

option = st.sidebar.selectbox(
    "Choose data source",
    ["Synthetic Signal", "Upload CSV"]
)

if option == "Synthetic Signal":
    t = np.linspace(0, 1, 500)
    signal = (
        np.sin(2 * np.pi * 5 * t)
        + 0.5 * np.sin(2 * np.pi * 20 * t)
        + 0.2 * np.random.randn(len(t))
    )
else:
    file = st.sidebar.file_uploader("Upload CSV with one column")
    if file:
        df = pd.read_csv(file)
        signal = df.iloc[:, 0].values
        t = np.arange(len(signal))
    else:
        st.stop()

# =========================
# Plot Raw Signal
# =========================
st.subheader("Raw Signal")
fig, ax = plt.subplots()
ax.plot(t, signal)
ax.set_title("Time Series")
st.pyplot(fig)

# =========================
# Feature Engineering Options
# =========================
st.sidebar.header("Feature Engineering")

use_fft = st.sidebar.checkbox("Fourier Transform", True)
use_wavelet = st.sidebar.checkbox("Wavelet Transform", True)

# =========================
# Fourier Features
# =========================
if use_fft:
    st.subheader("Fourier Transform Features")

    fft_vals = fft(signal)
    fft_magnitude = np.abs(fft_vals)

    fig, ax = plt.subplots()
    ax.plot(fft_magnitude)
    ax.set_title("FFT Magnitude Spectrum")
    st.pyplot(fig)

    dominant_freq = np.argmax(fft_magnitude[1:]) + 1
    spectral_energy = np.sum(fft_magnitude**2)

    st.write("Dominant Frequency Index:", dominant_freq)
    st.write("Spectral Energy:", spectral_energy)

# =========================
# Wavelet Features
# =========================
if use_wavelet:
    st.subheader("Wavelet Transform Features")

    wavelet = st.selectbox("Wavelet Type", ["db1", "db4", "sym5"])
    coeffs = pywt.wavedec(signal, wavelet)

    features = []
    for i, c in enumerate(coeffs):
        features.append({
            "level": i,
            "mean": np.mean(c),
            "std": np.std(c),
            "energy": np.sum(c**2)
        })

    wavelet_df = pd.DataFrame(features)
    st.dataframe(wavelet_df)

# =========================
# Feature Vector Summary
# =========================
st.subheader("Feature Vector Summary")

feature_vector = {}

if use_fft:
    feature_vector["fft_energy"] = spectral_energy
    feature_vector["fft_dominant_freq"] = dominant_freq

if use_wavelet:
    for i, c in enumerate(coeffs):
        feature_vector[f"wavelet_{i}_energy"] = np.sum(c**2)

st.json(feature_vector)
