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

# Debug info (bisa dihapus)
st.write("📁 Current Directory:", os.getcwd())
st.write("📄 Files:", os.listdir())

# ==========================================
# 0. LOAD DATA FIXED
# ==========================================
DATA_PATH = "PAPUABARAT2.xlsx"

try:
    df = pd.read_excel(DATA_PATH, sheet_name='Data Harian - Table')

    # hilangkan duplikasi nama kolom
    df = df.loc[:, ~df.columns.duplicated()]

    # mapping kecepatan angin
    if "kecepatan_angin" in df.columns:
        df = df.rename(columns={"kecepatan_angin": "FF_X"})

    # tanggal
    df['Tanggal'] = pd.to_datetime(df['Tanggal'], dayfirst=True)
    df['Tahun'] = df['Tanggal'].dt.year
    df['Bulan'] = df['Tanggal'].dt.month

    # ==========================================
    # Variabel yang digunakan
    # ==========================================
    possible_vars = ["Tn", "Tx", "Tavg", "kelembaban",
                     "curah_hujan", "matahari", "FF_X", "DDD_X"]
    available_vars = [v for v in possible_vars if v in df.columns]

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

    # ==========================================
    # 1. AGREGASI BULANAN
    # ==========================================
    agg_dict = {v: 'mean' for v in available_vars}
    if "curah_hujan" in available_vars:
        agg_dict["curah_hujan"] = "sum"

    cuaca_df = df[['Tahun', 'Bulan'] + available_vars]
    monthly_df = cuaca_df.groupby(['Tahun', 'Bulan']).agg(agg_dict).reset_index()

    st.subheader("📊 Data Bulanan Papua Barat")
    st.dataframe(monthly_df)

    # ==========================================
    # 2. TRAIN MODEL
    # ==========================================
    X = monthly_df[['Tahun', 'Bulan']]
    models = {}
    metrics = {}

    for var in available_vars:
        y = monthly_df[var]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        models[var] = model
        metrics[var] = {
            "rmse": np.sqrt(mean_squared_error(y_test, pred)),
            "r2": r2_score(y_test, pred)
        }

    # ==========================================
    # 3. METRIK
    # ==========================================
    st.subheader("📈 Evaluasi Model Machine Learning")
    for var, m in metrics.items():
        st.write(
            f"**{akademis_label[var]}** → RMSE: {m['rmse']:.3f} | R²: {m['r2']:.3f}"
        )

    # ==========================================
    # 4. PREDIKSI MANUAL
    # ==========================================
    st.subheader("🔮 Prediksi Manual (1 Bulan)")
    tahun_input = st.number_input("Masukkan Tahun Prediksi", min_value=2025, max_value=2100, value=2035)
    bulan_input = st.selectbox("Pilih Bulan", list(range(1, 13)))

    input_data = pd.DataFrame([[tahun_input, bulan_input]], columns=["Tahun", "Bulan"])

    st.write("### Hasil Prediksi:")
    for var in available_vars:
        pred_val = models[var].predict(input_data)[0]
        st.success(f"{akademis_label[var]} bulan {bulan_input}/{tahun_input}: **{pred_val:.2f}**")

    # ==========================================
    # 5. PREDIKSI OTOMATIS 2025–2075
    # ==========================================
    st.subheader("📆 Prediksi Otomatis 2025–2075")
    future_years = list(range(2025, 2076))
    future_months = list(range(1, 13))

    future_data = pd.DataFrame(
        [(year, month) for year in future_years for month in future_months],
        columns=['Tahun', 'Bulan']
    )

    for var in available_vars:
        future_data[f"Pred_{var}"] = models[var].predict(future_data[['Tahun', 'Bulan']])

    st.dataframe(future_data.head(12))

    # ==========================================
    # 6. GRAFIK HISTORIS VS PREDIKSI
    # ==========================================
    monthly_df['Sumber'] = 'Data Historis'
    future_data['Sumber'] = 'Prediksi'

    merge_list = []
    for var in available_vars:
        hist = monthly_df[['Tahun', 'Bulan', var, 'Sumber']].rename(columns={var: 'Nilai'})
        hist['Variabel'] = akademis_label[var]

        fut = future_data[['Tahun', 'Bulan', f"Pred_{var}", 'Sumber']].rename(columns={f"Pred_{var}": 'Nilai'})
        fut['Variabel'] = akademis_label[var]

        merge_list.append(pd.concat([hist, fut]))

    df_merge = pd.concat(merge_list)
    df_merge['Tanggal'] = pd.to_datetime(
        df_merge['Tahun'].astype(str) + "-" +
        df_merge['Bulan'].astype(str) + "-01"
    )

    st.subheader("📈 Grafik Tren Cuaca Papua Barat")
    selected_var = st.selectbox("Pilih Variabel Cuaca", [akademis_label[v] for v in available_vars])

    fig = px.line(
        df_merge[df_merge['Variabel'] == selected_var],
        x='Tanggal',
        y='Nilai',
        color='Sumber',
        title=f"Tren {selected_var} Bulanan",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ==========================================
    # 7. DOWNLOAD CSV
    # ==========================================
    st.subheader("💾 Simpan Hasil Prediksi")
    csv = future_data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download CSV Prediksi 2025–2075",
        data=csv,
        file_name='prediksi_papua_barat_2025_2075.csv',
        mime='text/csv'
    )

except FileNotFoundError:
    st.error("❌ File data belum ditemukan.\n\nPastikan file **PAPUABARAT2.xlsx** ada di folder aplikasi.")
