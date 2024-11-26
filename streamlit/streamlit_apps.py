import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import xgboost as xgb
import pandas as pd
import joblib
import os

# Memuat model yang sudah dilatih
model = xgb.Booster(model_file="/mount/src/narkolepsi-detector/model/xgb_narkolepsi.json")
X_train = joblib.load("/mount/src/narkolepsi-detector/model/train_data.pkl")

# UI Streamlit
st.set_page_config(page_title="Deteksi Narkolepsi", page_icon="🧠", layout="wide")

st.title("Deteksi Narkolepsi 🛌💤")

# Deskripsi yang lebih menarik
st.write(
    "Isi formulir berikut untuk mengetahui kemungkinan seseorang mengidap narkolepsi."
    " Data yang Anda masukkan akan digunakan untuk memberikan hasil prediksi yang akurat."
)

# Membuat input form dengan beberapa kategori
with st.form(key="narkolepsi_form"):
    st.header("Informasi Pengguna")

    usia = st.number_input("Usia", min_value=18, max_value=100, value=30)
    jenis_kelamin = st.selectbox("Jenis Kelamin", ["Pria", "Wanita"])
    riwayat_family_narkolepsi = st.selectbox(
        "Riwayat Narkolepsi di Keluarga", ["Tidak", "Ya"]
    )

    st.header("Kualitas Tidur & Gangguan Tidur")

    frekuensi_kantuk_siang = st.selectbox(
        "Frekuensi Kantuk Siang", ["Jarang", "Kadang-kadang", "Sering", "Sangat sering"]
    )
    pengalaman_katapleksi = st.selectbox("Pengalaman Katapleksi", ["Tidak", "Ya"])
    pengalaman_hallusinasi_tidur = st.selectbox(
        "Pengalaman Hallusinasi Tidur", ["Tidak", "Ya"]
    )
    pengalaman_paralisis_tidur = st.selectbox(
        "Pengalaman Paralisis Tidur", ["Tidak", "Ya"]
    )

    st.header("Durasi & Kualitas Tidur")

    durasi_tidur_malam = st.number_input(
        "Durasi Tidur Malam (jam)", min_value=4, max_value=12, value=7
    )
    kualitas_tidur_malam = st.selectbox(
        "Kualitas Tidur Malam", ["Buruk", "Sedang", "Baik"]
    )
    durasi_tidur_siang = st.number_input(
        "Durasi Tidur Siang (menit)", min_value=0, max_value=180, value=30
    )

    st.header("Faktor Lainnya")

    frekuensi_gangguan_sleep = st.selectbox("Frekuensi Gangguan Tidur", ["Tidak", "Ya"])
    riwayat_gangguan_medis = st.selectbox("Riwayat Gangguan Medis", ["Tidak", "Ya"])
    tes_sleep_latency_mslt = st.number_input(
        "Tes Sleep Latency MSLT (menit)", min_value=0.0, max_value=30.0, value=5.0
    )
    kantuk_dipicu_emosi = st.selectbox("Kantuk Dipicu Emosi", ["Tidak", "Ya"])

    submit_button = st.form_submit_button("Prediksi Narkolepsi")


# Fungsi prediksi
def make_prediction(inputs):
    # Membuat dataframe untuk inputan pengguna
    df = pd.DataFrame(
        [inputs],
        columns=[
            "Usia",
            "Jenis_Kelamin",
            "Riwayat_Family_Narkolepsi",
            "Frekuensi_Kantuk_Siang",
            "Pengalaman_Katapleksi",
            "Pengalaman_Hallusinasi_Tidur",
            "Pengalaman_Paralisis_Tidur",
            "Durasi_Tidur_Malam",
            "Kualitas_Tidur_Malam",
            "Durasi_Tidur_Siang",
            "Frekuensi_Gangguan_Sleep",
            "Riwayat_Gangguan_Medis",
            "Tes_Sleep_Latency_MSLT",
            "Kantuk_Dipicu_Emosi",
        ],
    )

    # Encoding untuk kolom kategorikal
    df["Jenis_Kelamin"] = (
        df["Jenis_Kelamin"]
        .astype("category")
        .cat.set_categories(X_train["Jenis_Kelamin"].cat.categories)
    )
    df["Riwayat_Family_Narkolepsi"] = (
        df["Riwayat_Family_Narkolepsi"]
        .astype("category")
        .cat.set_categories(X_train["Riwayat_Family_Narkolepsi"].cat.categories)
    )
    df["Frekuensi_Kantuk_Siang"] = (
        df["Frekuensi_Kantuk_Siang"]
        .astype("category")
        .cat.set_categories(X_train["Frekuensi_Kantuk_Siang"].cat.categories)
    )
    df["Pengalaman_Katapleksi"] = (
        df["Pengalaman_Katapleksi"]
        .astype("category")
        .cat.set_categories(X_train["Pengalaman_Katapleksi"].cat.categories)
    )
    df["Pengalaman_Hallusinasi_Tidur"] = (
        df["Pengalaman_Hallusinasi_Tidur"]
        .astype("category")
        .cat.set_categories(X_train["Pengalaman_Hallusinasi_Tidur"].cat.categories)
    )
    df["Pengalaman_Paralisis_Tidur"] = (
        df["Pengalaman_Paralisis_Tidur"]
        .astype("category")
        .cat.set_categories(X_train["Pengalaman_Paralisis_Tidur"].cat.categories)
    )
    df["Kualitas_Tidur_Malam"] = (
        df["Kualitas_Tidur_Malam"]
        .astype("category")
        .cat.set_categories(X_train["Kualitas_Tidur_Malam"].cat.categories)
    )
    df["Frekuensi_Gangguan_Sleep"] = (
        df["Frekuensi_Gangguan_Sleep"]
        .astype("category")
        .cat.set_categories(X_train["Frekuensi_Gangguan_Sleep"].cat.categories)
    )
    df["Riwayat_Gangguan_Medis"] = (
        df["Riwayat_Gangguan_Medis"]
        .astype("category")
        .cat.set_categories(X_train["Riwayat_Gangguan_Medis"].cat.categories)
    )
    df["Kantuk_Dipicu_Emosi"] = (
        df["Kantuk_Dipicu_Emosi"]
        .astype("category")
        .cat.set_categories(X_train["Kantuk_Dipicu_Emosi"].cat.categories)
    )

    # Menggunakan predict_proba() untuk mendapatkan probabilitas
    pred_proba = model.predict(xgb.DMatrix(df, enable_categorical=True))
    return float(pred_proba[0])  # Mengembalikan probabilitas untuk kelas 1 (narkolepsi)


# Prediksi jika tombol ditekan
if submit_button:
    # Mengumpulkan data input dari form
    inputs = {
        "Usia": usia,
        "Jenis_Kelamin": jenis_kelamin,
        "Riwayat_Family_Narkolepsi": riwayat_family_narkolepsi,
        "Frekuensi_Kantuk_Siang": frekuensi_kantuk_siang,
        "Pengalaman_Katapleksi": pengalaman_katapleksi,
        "Pengalaman_Hallusinasi_Tidur": pengalaman_hallusinasi_tidur,
        "Pengalaman_Paralisis_Tidur": pengalaman_paralisis_tidur,
        "Durasi_Tidur_Malam": durasi_tidur_malam,
        "Kualitas_Tidur_Malam": kualitas_tidur_malam,
        "Durasi_Tidur_Siang": durasi_tidur_siang,
        "Frekuensi_Gangguan_Sleep": frekuensi_gangguan_sleep,
        "Riwayat_Gangguan_Medis": riwayat_gangguan_medis,
        "Tes_Sleep_Latency_MSLT": tes_sleep_latency_mslt,
        "Kantuk_Dipicu_Emosi": kantuk_dipicu_emosi,
    }

    proba_pos = make_prediction(inputs)

    # Menampilkan hasil probabilitas dengan format persentase
    proba_pos_percentage = proba_pos * 100
    proba_neg_percentage = (1 - proba_pos) * 100

    st.subheader(f"Probabilitas Deteksi Narkolepsi:")
    st.markdown(f"**Tidak menderita Narkolepsi:** {proba_neg_percentage:.2f}%")
    st.markdown(f"**Menderita Narkolepsi:** {proba_pos_percentage:.2f}%")

    # Visualisasi grafik batang
    fig, ax = plt.subplots()
    ax.bar(
        ["Tidak Narkolepsi", "Narkolepsi"],
        [proba_neg_percentage, proba_pos_percentage],
        color=["#3498db", "#e74c3c"],
    )
    ax.set_ylabel("Probabilitas (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Probabilitas Narkolepsi")
    st.pyplot(fig)

    # Memberikan hasil prediksi berdasarkan probabilitas
    if proba_pos > 0.5:
        st.success("Anda kemungkinan besar menderita Narkolepsi.")
    else:
        st.success("Anda kemungkinan besar tidak menderita Narkolepsi.")
