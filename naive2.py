import os
import numpy as np
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Fungsi memuat data
@st.cache_data
def load_or_create_dataframe(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    elif 'DataMurni.CSV' in os.listdir():
        df = pd.read_csv('DataMurni.CSV')
    else:
        df = pd.DataFrame()  # Data kosong
    return df

# ===================== APLIKASI UTAMA =====================
st.set_page_config(page_title="Deteksi Masalah Android - Naive Bayes")
st.sidebar.title("Navigasi")
selected_option = st.sidebar.radio("Pilih Halaman", [
    "Pratinjau Data", "Tambah Data Baru", "Deteksi Sekarang", "Latih Model Naive Bayes", "Langkah-Langkah Perhitungan"])

st.title("Deteksi Mandiri Masalah Perangkat Lunak Smartphone Android")

uploaded_file = st.file_uploader("Silakan Upload File CSV", type=["csv"])
df = load_or_create_dataframe(uploaded_file)

# Inisialisasi model
if 'model' not in st.session_state:
    st.session_state.model = None
    st.session_state.X_test = None

# ===================== PRATINJAU DATA =====================
if selected_option == "Pratinjau Data":
    st.subheader("📊 Pratinjau Data")
    if not df.empty:
        st.dataframe(df)
    else:
        st.warning("Data belum tersedia.")

# ===================== TAMBAH DATA BARU =====================
elif selected_option == "Tambah Data Baru":
    if not df.empty:
        with st.form("Tambah Data"):
            st.write("Isi Form Berikut:")
            new_data = {}
            for col in df.columns:
                if col != 'Masalah':
                    new_data[col] = st.selectbox(f"{col}", options=df[col].unique(), key=f"input_{col}")

            if st.form_submit_button("Simpan Data"):
                try:
                    features = [col for col in df.columns if col != 'Masalah']
                    X = df[features]
                    y = df['Masalah']

                    encoders = {col: LabelEncoder().fit(X[col]) for col in X.columns}
                    for col, encoder in encoders.items():
                        X[col] = encoder.transform(X[col])

                    y_encoder = LabelEncoder().fit(y)
                    y = y_encoder.transform(y)

                    model = CategoricalNB()
                    model.fit(X, y)

                    new_data_encoded = {col: encoders[col].transform([new_data[col]])[0] for col in new_data}
                    new_prediction = model.predict(pd.DataFrame([new_data_encoded]))
                    new_prediction_decoded = y_encoder.inverse_transform(new_prediction)

                    new_data['Masalah'] = new_prediction_decoded[0]
                    new_data_df = pd.DataFrame([new_data])
                    df = pd.concat([df, new_data_df], ignore_index=True)
                    df.to_csv('DataMurni.CSV', index=False)
                    st.success("Data berhasil ditambahkan.")
                except Exception as e:
                    st.error(f"Kesalahan: {e}")
    else:
        st.warning("Upload data terlebih dahulu.")

# ===================== DETEKSI SEKARANG =====================
elif selected_option == "Deteksi Sekarang":
    if not df.empty:
        new_data = {}
        with st.form("Deteksi Sekarang"):
            for col in df.columns:
                if col != 'Masalah':
                    new_data[col] = st.selectbox(f"{col}", options=df[col].unique(), key=f"det_{col}")

            if st.form_submit_button("Deteksi"):
                try:
                    features = [col for col in df.columns if col != 'Masalah']
                    X = df[features]
                    y = df['Masalah']

                    encoders = {col: LabelEncoder().fit(X[col]) for col in X.columns}
                    for col, encoder in encoders.items():
                        X[col] = encoder.transform(X[col])

                    y_encoder = LabelEncoder().fit(y)
                    y = y_encoder.transform(y)

                    model = CategoricalNB()
                    model.fit(X, y)

                    new_data_encoded = {col: encoders[col].transform([new_data[col]])[0] for col in new_data}
                    new_prediction = model.predict(pd.DataFrame([new_data_encoded]))
                    prediction_result = y_encoder.inverse_transform(new_prediction)[0]

                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    st.write(f"Akurasi model: {accuracy * 100:.2f}%")

                    st.markdown(f"### Hasil Deteksi: {prediction_result}")

                except Exception as e:
                    st.error(f"Kesalahan: {e}")
    else:
        st.warning("Upload data terlebih dahulu.")

# ===================== LATIH MODEL =====================
elif selected_option == "Latih Model Naive Bayes":
    if not df.empty:
        with st.form("Latih Model"):
            target = st.selectbox("Pilih Target", options=df.columns.tolist(), index=len(df.columns)-1)
            features = st.multiselect("Pilih Fitur", options=[col for col in df.columns if col != target], default=[col for col in df.columns if col != target])
            test_size = st.slider("Persentase Data Uji", min_value=10, max_value=50, value=20, step=5) / 100

            if st.form_submit_button("Latih Model"):
                X = df[features]
                y = df[target]

                encoders = {col: LabelEncoder().fit(X[col]) for col in X.columns}
                for col, encoder in encoders.items():
                    X[col] = encoder.transform(X[col])

                y_encoder = LabelEncoder().fit(y)
                y = y_encoder.transform(y)

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                model = CategoricalNB()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                st.session_state.model = model
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
                st.session_state.y_pred = y_pred
                st.session_state.y_encoder = y_encoder

                st.write(f"Akurasi: {accuracy_score(y_test, y_pred) * 100:.2f}%")

                conf_matrix = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax)
                st.pyplot(fig)
    else:
        st.warning("Upload data terlebih dahulu.")

# ===================== LANGKAH-LANGKAH =====================
elif selected_option == "Langkah-Langkah Perhitungan":
    st.markdown("## Langkah-Langkah Perhitungan Naive Bayes")
    if st.session_state.model:
        model = st.session_state.model
        class_prob = np.exp(model.class_log_prior_)
        st.write("### Probabilitas A Priori:", class_prob)

        st.write("### Probabilitas Fitur Bersyarat:")
        for i, feature_probs in enumerate(model.feature_log_prob_):
            st.write(f"Fitur ke-{i}:", np.exp(feature_probs))

        st.write("### Probabilitas Akhir (contoh data uji):")
        if st.session_state.X_test is not None:
            final_log_probs = model.predict_log_proba(st.session_state.X_test)
            for i, probs in enumerate(final_log_probs[:3]):
                st.write(f"Data uji {i}:", np.exp(probs))
    else:
        st.warning("Model belum dilatih.")