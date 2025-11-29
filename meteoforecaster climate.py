import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os


# ==========================
# HEADER
# ==========================

st.markdown("""
<h1 style='text-align:center; color:#0076B6;'>🌦️ Prediksi Iklim Wilayah Papua Barat</h1>
<p style='text-align:center; font-size:17px; color:#555;'>Analisis berdasarkan data historis cuaca dan prediksi machine learning hingga tahun 2075.</p>
<hr style='border:1px solid #0076B6;'>
""", unsafe_allow_html=True)


# ==========================
# LOAD DATA
# ==========================

DATA_PATH = "PAPUABARAT2.xlsx"

with st.sidebar:
    st.header("📁 File Status")
    st.write(os.listdir())
    st.info("Menggunakan dataset: PAPUABARAT2.xlsx")
    st.markdown("---")


try:
    df = pd.read_excel(DATA_PATH, sheet_name='Data Harian - Table')
    df = df.loc[:, ~df.columns.duplicated()]

    if "kecepatan_angin" in df.columns:
        df.rename(columns={"kecepatan_angin": "FF_X"}, inplace=True)

    df['Tanggal'] = pd.to_datetime(df['Tanggal'], dayfirst=True)
    df['Tahun'] = df['Tanggal'].dt.year
    df['Bulan'] = df['Tanggal'].dt.month

    possible_vars = ["Tn", "Tx", "Tavg", "kelembaban", "curah_hujan", "matahari", "FF_X", "DDD_X"]
    available_vars = [v for v in possible_vars if v in df.columns]

    akademis_label = {
        "Tn": "🌡️ Suhu Minimum (°C)",
        "Tx": "🔥 Suhu Maksimum (°C)",
        "Tavg": "🌥️ Suhu Rata-rata (°C)",
        "kelembaban": "💧 Kelembaban Udara (%)",
        "curah_hujan": "🌧️ Curah Hujan (mm)",
        "matahari": "☀️ Durasi Penyinaran Matahari (jam)",
        "FF_X": "💨 Kecepatan Angin Maksimum (m/s)",
        "DDD_X": "🧭 Arah Angin (°)"
    }

    # ==========================
    # AGREGASI BULANAN
    # ==========================
    agg_dict = {v: 'mean' for v in available_vars}
    if "curah_hujan" in available_vars:
        agg_dict["curah_hujan"] = "sum"

    monthly_df = df.groupby(['Tahun', 'Bulan']).agg(agg_dict).reset_index()

    st.subheader("📊 Data Cuaca Bulanan Papua Barat")
    st.dataframe(monthly_df.style.highlight_max(axis=0, color="#C2F2FF"))

    # ==========================
    # TRAIN MODEL
    # ==========================
    X = monthly_df[['Tahun', 'Bulan']]
    models, metrics = {}, {}

    for var in available_vars:
        y = monthly_df[var]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        models[var] = model
        metrics[var] = {"rmse": np.sqrt(mean_squared_error(y_test, pred)), "r2": r2_score(y_test, pred)}

    st.subheader("📈 Evaluasi Model")
    for var, m in metrics.items():
        st.write(f"**{akademis_label[var]}** → RMSE: `{m['rmse']:.3f}` | R²: `{m['r2']:.3f}`")

    # ==========================
    # PREDIKSI MANUAL
    # ==========================

    st.subheader("🔍 Prediksi Manual")
    col1, col2 = st.columns(2)

    tahun_input = col1.number_input("Pilih Tahun Prediksi:", min_value=2025, max_value=2100, value=2035)
    bulan_input = col2.selectbox("Pilih Bulan:", list(range(1, 13)))

    input_data = pd.DataFrame([[tahun_input, bulan_input]], columns=["Tahun", "Bulan"])

    st.write("### Hasil Prediksi Bulan Ini:")

    for var in available_vars:
        pred = models[var].predict(input_data)[0]
        st.success(f"{akademis_label[var]}: **{pred:.2f}**")


    # ==========================
    # PREDIKSI 2025–2075
    # ==========================

    st.subheader("📆 Prediksi Otomatis 2025–2075")

    future_data = pd.DataFrame([(y, m) for y in range(2025, 2076) for m in range(1, 13)],
                                columns=["Tahun", "Bulan"])

    for var in available_vars:
        future_data[f"Pred_{var}"] = models[var].predict(future_data[['Tahun', 'Bulan']])

    st.dataframe(future_data.head(12))


    # ==========================
    # VISUALISASI GRAFIK
    # ==========================

    st.subheader("📈 Grafik Perubahan Iklim Papua Barat")

    future_data['Sumber'] = 'Prediksi'
    monthly_df['Sumber'] = 'Historis'

    merged = []
    for var in available_vars:
        hist = monthly_df[['Tahun', 'Bulan', var, 'Sumber']].rename(columns={var: "Nilai"})
        hist['Variabel'] = akademis_label[var]

        fut = future_data[['Tahun', 'Bulan', f"Pred_{var}", 'Sumber']].rename(columns={f"Pred_{var}": "Nilai"})
        fut['Variabel'] = akademis_label[var]

        merged.append(pd.concat([hist, fut]))

    df_plot = pd.concat(merged)
    df_plot['Tanggal'] = pd.to_datetime(df_plot['Tahun'].astype(str) + "-" + df_plot['Bulan'].astype(str) + "-01")

    pilih_var = st.selectbox("Pilih Variabel:", [akademis_label[v] for v in available_vars])

    fig = px.line(df_plot[df_plot['Variabel'] == pilih_var], x='Tanggal', y='Nilai', color='Sumber',
                  title=f"📌 Tren {pilih_var} dari Waktu ke Waktu")

    st.plotly_chart(fig, use_container_width=True)


    # ==========================
    # DOWNLOAD
    # ==========================

    st.subheader("💾 Simpan Hasil")
    st.download_button("📥 Download CSV Prediksi", future_data.to_csv(index=False),
                       file_name="Prediksi_PapuaBarat_2025_2075.csv")

except FileNotFoundError:
    st.error("❌ File PAPUABARAT2.xlsx tidak ditemukan. Simpan file di folder yang sama.")



