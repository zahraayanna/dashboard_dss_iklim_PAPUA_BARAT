import streamlit as st 
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os

# Judul Dashboard
st.title("🌦️ Prediksi Iklim Papua Barat (Tanpa Upload Data)")
st.write("Dashboard otomatis menggunakan data historis Papua Barat.")

# Path file dataset
DATA_PATH = "PAPUABARAT2.xlsx"

# Debug (opsional)
st.write("📂 Folder berisi file:", os.listdir())

try:
    # Load data
    df = pd.read_excel(DATA_PATH, sheet_name='Data Harian - Table')

    # Hilangkan duplikasi kolom
    df = df.loc[:, ~df.columns.duplicated()]

    # Normalisasi nama kolom angin
    if "kecepatan_angin" in df.columns:
        df.rename(columns={"kecepatan_angin": "FF_X"}, inplace=True)

    # Konversi tanggal
    df['Tanggal'] = pd.to_datetime(df['Tanggal'], dayfirst=True)
    df['Tahun'] = df['Tanggal'].dt.year
    df['Bulan'] = df['Tanggal'].dt.month

    # Variabel yang digunakan
    possible_vars = ["Tn", "Tx", "Tavg", "kelembaban", "curah_hujan", "matahari", "FF_X", "DDD_X"]
    available_vars = [v for v in possible_vars if v in df.columns]

    # Label akademis
    akademis_label = {
        "Tn": "Suhu Minimum (°C)",
        "Tx": "Suhu Maksimum (°C)",
        "Tavg": "Suhu Rata-rata (°C)",
        "kelembaban": "Kelembaban Udara (%)",
        "curah_hujan": "Curah Hujan (mm)",
        "matahari": "Durasi Penyinaran Matahari (jam)",
        "FF_X": "Kecepatan Angin Maksimum (m/s)",
        "DDD_X": "Arah Angin saat Kecepatan Maksimum (°)"
    }

    # Agregasi Bulanan
    agg_dict = {v: 'mean' for v in available_vars}
    if "curah_hujan" in available_vars:
        agg_dict["curah_hujan"] = "sum"

    monthly_df = df.groupby(['Tahun', 'Bulan']).agg(agg_dict).reset_index()

    st.subheader("📊 Data Bulanan Papua Barat")
    st.dataframe(monthly_df)

    # Model Training
    X = monthly_df[['Tahun', 'Bulan']]
    models = {}
    metrics = {}

    for var in available_vars:
        y = monthly_df[var]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        models[var] = model
        metrics[var] = {"rmse": np.sqrt(mean_squared_error(y_test, pred)), "r2": r2_score(y_test, pred)}

    st.subheader("📈 Evaluasi Model Machine Learning")
    for var, m in metrics.items():
        st.write(f"**{akademis_label[var]}** → RMSE: {m['rmse']:.3f} | R²: {m['r2']:.3f}")

    # Prediksi manual
    st.subheader("🔍 Prediksi Berdasarkan Input Bulan & Tahun")
    tahun_input = st.number_input("Masukkan Tahun:", min_value=2025, max_value=2100, value=2035)
    bulan_input = st.selectbox("Pilih Bulan", list(range(1, 12)))

    input_data = pd.DataFrame([[tahun_input, bulan_input]], columns=["Tahun", "Bulan"])

    st.write("### Hasil Prediksi:")
    for var in available_vars:
        pred = models[var].predict(input_data)[0]
        st.success(f"{akademis_label[var]} {bulan_input}/{tahun_input}: **{pred:.2f}**")

    # Prediksi otomatis 2025–2075
    st.subheader("📆 Prediksi Otomatis 2025–2075")
    future_data = pd.DataFrame([(y, m) for y in range(2025, 2076) for m in range(1, 13)],
                                columns=["Tahun", "Bulan"])

    for var in available_vars:
        future_data[f"Pred_{var}"] = models[var].predict(future_data[['Tahun', 'Bulan']])

    st.dataframe(future_data.head(12))

    # Grafik tren
    monthly_df['Sumber'] = 'Data Historis'
    future_data['Sumber'] = 'Prediksi'

    merged = []
    for var in available_vars:
        hist = monthly_df[['Tahun', 'Bulan', var, 'Sumber']].rename(columns={var: "Nilai"})
        hist['Variabel'] = akademis_label[var]

        fut = future_data[['Tahun', 'Bulan', f"Pred_{var}", 'Sumber']].rename(columns={f"Pred_{var}": "Nilai"})
        fut['Variabel'] = akademis_label[var]

        merged.append(pd.concat([hist, fut]))

    df_plot = pd.concat(merged)
    df_plot['Tanggal'] = pd.to_datetime(df_plot['Tahun'].astype(str) + "-" + df_plot['Bulan'].astype(str) + "-01")

    st.subheader("📈 Grafik Tren Iklim Papua Barat")
    pilih_var = st.selectbox("Pilih Variabel", [akademis_label[v] for v in available_vars])

    fig = px.line(df_plot[df_plot['Variabel'] == pilih_var], x='Tanggal', y='Nilai',
                  color='Sumber', title=f"Tren Perubahan {pilih_var}")

    st.plotly_chart(fig, use_container_width=True)

    # Download hasil prediksi
    st.subheader("💾 Download Hasil Prediksi")
    st.download_button("📥 Download Data CSV", future_data.to_csv(index=False), 
                       file_name="Prediksi_PapuaBarat_2025_2075.csv")

except FileNotFoundError:
    st.error("❌ File PAPUABARAT2.xlsx tidak ditemukan.")

