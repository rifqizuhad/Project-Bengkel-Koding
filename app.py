import streamlit as st
import pickle
import numpy as np

# ------------------------------#
# Konfigurasi halaman
# ------------------------------#
st.set_page_config(page_title="Prediksi Obesitas", layout="centered")
st.title("🍔 Prediksi Tingkat Obesitas dengan Random Forest")

# ------------------------------#
# Fungsi bantu
# ------------------------------#
@st.cache_resource
def load_pickle_file(path: str):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error(f"File '{path}' tidak ditemukan.")
        return None

def build_feature_array(inputs: dict) -> np.ndarray:
    ordered_keys = [
        "Age", "Gender", "Height", "Weight", "CALC", "FAVC", "FCVC",
        "NCP", "SCC", "SMOKE", "CH2O", "family_history_with_overweight",
        "FAF", "TUE", "CAEC", "MTRANS"
    ]
    return np.array([[inputs[k] for k in ordered_keys]])

# ------------------------------#
# Load model & scaler
# ------------------------------#
model = load_pickle_file("model_random_forest.pkl")
scaler = load_pickle_file("scaler.pkl")

# Label map
label_map = {
    0: "Insufficient_Weight",
    1: "Normal_Weight",
    2: "Obesity_Type_I",
    3: "Obesity_Type_II",
    4: "Obesity_Type_III",
    5: "Overweight_Level_I",
    6: "Overweight_Level_II"
}

# ------------------------------#
# Form input user
# ------------------------------#
with st.form("form_prediksi"):
    st.subheader("📋 Masukkan Data Anda")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Usia", 0, 100, 25)
        gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
        height = st.number_input("Tinggi Badan (m)", 1.0, 2.5, 1.70)
        weight = st.number_input("Berat Badan (kg)", 1, 200, 60)
        calc = st.selectbox("Konsumsi Alkohol", ["no", "Sometimes", "Frequently", "Always"])
        favc = st.selectbox("Sering makan tinggi kalori?", ["yes", "no"])
        fcvc = st.slider("Frekuensi makan sayur (1-3)", 1.0, 3.0, 2.0)

    with col2:
        ncp = st.slider("Makan utama per hari", 1.0, 4.0, 3.0)
        scc = st.selectbox("Pantau kalori?", ["yes", "no"])
        smoke = st.selectbox("Merokok?", ["yes", "no"])
        ch2o = st.slider("Air putih (liter/hari)", 1.0, 3.0, 2.0)
        fam_hist = st.selectbox("Riwayat obesitas keluarga?", ["yes", "no"])
        faf = st.slider("Aktivitas fisik (jam/minggu)", 0.0, 3.0, 1.0)
        tue = st.slider("Waktu layar (jam/hari)", 0.0, 2.0, 1.0)
        caec = st.selectbox("Kebiasaan ngemil", ["no", "Sometimes", "Frequently", "Always"])
        mtrans = st.selectbox("Transportasi", ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"])

    submitted = st.form_submit_button("🔍 Prediksi")

# ------------------------------#
# Proses Prediksi
# ------------------------------#
if submitted and model and scaler:
    input_dict = {
        "Age": age,
        "Gender": 1 if gender == "Male" else 0,
        "Height": height,
        "Weight": weight,
        "CALC": ["no", "Sometimes", "Frequently", "Always"].index(calc),
        "FAVC": 1 if favc == "yes" else 0,
        "FCVC": fcvc,
        "NCP": ncp,
        "SCC": 1 if scc == "yes" else 0,
        "SMOKE": 1 if smoke == "yes" else 0,
        "CH2O": ch2o,
        "family_history_with_overweight": 1 if fam_hist == "yes" else 0,
        "FAF": faf,
        "TUE": tue,
        "CAEC": ["no", "Sometimes", "Frequently", "Always"].index(caec),
        "MTRANS": ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"].index(mtrans),
    }

    feature_array = build_feature_array(input_dict)
    scaled_features = scaler.transform(feature_array)

    pred = model.predict(scaled_features)[0]
    st.success(f"Hasil Prediksi: **{label_map[pred]}**")
