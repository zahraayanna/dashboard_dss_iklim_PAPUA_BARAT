import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# ==============================
# CONFIG TAMPILAN
# ==============================
st.set_page_config(
    page_title="📊 Dashboard DSS Iklim | Papua Barat",
    layout="wide"
)

st.markdown(
    """
    <h1 style="text-align:center; color:#0A5FA4;">📍 Dashboard Analisis & Prediksi Iklim<br>Provinsi Papua Barat</h1>
    <hr>
    """,
    unsafe_allow_html=True
)

# ==============================
# LOAD DATA
# ==============================
FILE_NAME = "PAPUABARAT2.xlsx"

try:
    df = pd.read_excel(FILE_NAME)
    st.success("📁 Data berhasil dimuat!")
except:
    st.error(f"❌ File **{FILE_NAME}** tidak ditemukan. Pastikan file berada di folder aplikasi.")
    st.stop()

# ==============================
# SIDEBAR
# ==============================
st.sidebar.header("⚙️ Pengaturan Tampilan")
selected_column = st.sidebar.selectbox("Pilih Variabel Iklim:", df.columns[1:], index=0)
moving_avg_window = st.sidebar.slider("Moving Average (Periode):", 2, 12, 3)

# ==============================
# TAMPILKAN DATA
# ==============================
with st.expander("📌 Lihat Data Mentah"):
    st.dataframe(df)

# ==============================
# GRAFIK UTAMA (PLOTLY)
# ==============================
st.subheader(f"📈 Tren {selected_column} dari Tahun ke Tahun")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df[df.columns[0]], 
    y=df[selected_column],
    mode='lines+markers',
    line=dict(width=3),
    name="Data Asli"
))

# moving average
df["MA"] = df[selected_column].rolling(window=moving_avg_window).mean()

fig.add_trace(go.Scatter(
    x=df[df.columns[0]],
    y=df["MA"],
    mode='lines',
    line=dict(width=4, dash="dash"),
    name=f"Moving Average ({moving_avg_window})"
))

fig.update_layout(
    template="simple_white",
    height=450,
    title=f"📊 Visualisasi Tren {selected_column}",
    xaxis_title="Tahun",
    yaxis_title=selected_column,
    showlegend=True
)

st.plotly_chart(fig, use_container_width=True)

# ==============================
# FORECAST SEDERHANA
# ==============================
st.subheader("🔮 Prediksi Sederhana 5 Tahun Ke Depan")

# model prediksi linear sederhana
x = np.arange(len(df))
y = df[selected_column].values

m, b = np.polyfit(x, y, 1)

future_years = np.arange(len(df), len(df) + 5)
future_predictions = m * future_years + b

future_df = pd.DataFrame({
    "Tahun": range(df[df.columns[0]].max() + 1, df[df.columns[0]].max() + 6),
    f"Prediksi {selected_column}": future_predictions
})

st.dataframe(future_df)

# ==============================
# DOWNLOAD HASIL
# ==============================
st.subheader("📥 Download Hasil Analisis")

if st.button("📄 Export ke Excel"):
    export_df = pd.concat([df, future_df], ignore_index=True)
    export_df.to_excel("Hasil_Analisis_PapuaBarat.xlsx", index=False)
    st.success("✔ File berhasil diekspor sebagai: `Hasil_Analisis_PapuaBarat.xlsx`")

# ==============================
# FOOTER
# ==============================
st.markdown(
    """
    <br><hr>
    <p style="text-align:center; color:gray;">
    🌿 Dikembangkan untuk sistem pendukung keputusan iklim Papua Barat | Streamlit © 2025
    </p>
    """,
    unsafe_allow_html=True
)
