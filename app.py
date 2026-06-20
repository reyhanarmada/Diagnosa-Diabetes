import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Diagnosa Diabetes", page_icon="🩺", layout="centered")

# ---------- Load artifacts ----------
@st.cache_resource
def load_artifacts():
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('model_rf.pkl', 'rb') as f:
        model_rf = pickle.load(f)
    with open('feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    return scaler, model_rf, feature_names

scaler, model_rf, feature_names = load_artifacts()

# ---------- UI ----------
st.title("🩺 Sistem Diagnosa Diabetes")
st.markdown("**Kelompok 3 - Mata Kuliah Data Mining**")
st.markdown("Masukkan data pasien untuk memprediksi kemungkinan diabetes berdasarkan model klasifikasi.")

st.divider()

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies (jumlah kehamilan)", min_value=0, max_value=20, value=1, step=1)
    glucose = st.number_input("Glucose (mg/dL)", min_value=0, max_value=300, value=120)
    blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70)
    skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)

with col2:
    insulin = st.number_input("Insulin (mu U/ml)", min_value=0, max_value=900, value=80)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
    age = st.number_input("Age (tahun)", min_value=1, max_value=120, value=30)

st.divider()

if st.button("🔍 Prediksi", use_container_width=True, type="primary"):
    input_dict = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }

    input_df = pd.DataFrame([input_dict])[feature_names]
    input_scaled = scaler.transform(input_df)

    prediction = model_rf.predict(input_scaled)[0]
    proba = model_rf.predict_proba(input_scaled)[0]

    st.divider()
    if prediction == 1:
        st.error("### Hasil: Berisiko Diabetes (Outcome = 1)")
    else:
        st.success("### Hasil: Tidak Berisiko Diabetes (Outcome = 0)")

    c1, c2 = st.columns(2)
    c1.metric("Probabilitas Tidak Diabetes", f"{proba[0]*100:.2f}%")
    c2.metric("Probabilitas Diabetes", f"{proba[1]*100:.2f}%")

    st.caption("Model digunakan: **Random Forest**")
    st.caption("⚠️ Hasil ini hanya untuk tujuan edukasi/akademik, bukan diagnosis medis resmi.")

st.divider()
with st.expander("ℹ️ Tentang Model"):
    st.markdown("""
    - **Random Forest** dipilih sebagai model terbaik berdasarkan hasil evaluasi setelah proses preprocessing, SMOTE, dan optimasi hyperparameter menggunakan GridSearchCV.
    - Hyperparameter terbaik yang diperoleh adalah:
      - n_estimators = 300
      - max_depth = None
      - min_samples_split = 2
      - min_samples_leaf = 1
    - F1-Score (macro): **0.8626**
    - Perbandingan performa model:
      - Logistic Regression: 0.7399
      - Random Forest: 0.8626
      - XGBoost: 0.8543
    - Random Forest dipilih karena menghasilkan nilai F1-Score Macro tertinggi serta memiliki kemampuan yang baik dalam menangani hubungan non-linear dan mengurangi risiko overfitting melalui mekanisme ensemble learning.
    - Data preprocessing meliputi penggantian nilai 0 pada atribut medis (Glucose, BloodPressure, SkinThickness, Insulin, dan BMI) menggunakan median, dilanjutkan dengan standardisasi fitur menggunakan StandardScaler.
    - Teknik SMOTE digunakan untuk menyeimbangkan distribusi kelas pada data latih sebelum proses pelatihan model.
    - Dataset yang digunakan adalah Pima Indians Diabetes Dataset.
    """)