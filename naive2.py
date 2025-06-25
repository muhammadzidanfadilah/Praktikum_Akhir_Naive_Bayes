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

# Fungsi untuk cek Login
@st.cache_data
def load_login_data():
    if 'login.csv' in os.listdir():
        login_df = pd.read_csv('login.csv')
        return login_df
    else:
        return pd.DataFrame(columns=['username', 'password', 'role'])

def check_login(username, password, login_df):
    user_record = login_df[(login_df['username'] == username) & (login_df['password'] == password)]
    if not user_record.empty:
        return user_record['role'].values[0]
    else:
        return None

# Fungsi logout
def logout():
    # Menghapus semua data dari session_state
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state.logged_in = False
    st.session_state.user_level = None
    st.success("Anda Berhasil Keluar!")

# Memuat atau membuat fungsi dataframe
@st.cache_data
def load_or_create_dataframe(uploaded_file, user_level):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    elif user_level == "user":
        if 'DataMurni.CSV' in os.listdir():
            df = pd.read_csv('DataMurni.CSV')
        else:
            df = pd.DataFrame()  # DataFrame kosong tanpa kolom
    else:
        df = pd.DataFrame()  # DataFrame kosong tanpa kolom
    return df

# Fungsi untuk mengonversi DataFrame ke CSV
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# Membuat data login
login_df = load_login_data()

# Halaman Login
if 'logged_in' not in st.session_state:
    st.title("Silahkan Masuk Untuk Deteksi Mandiri Masalah Perangkat Lunak Smartphone Android")
    st.session_state.logged_in = False
    st.session_state.user_level = None

if not st.session_state.logged_in:
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        user_level = check_login(username, password, login_df)
        if user_level:
            st.session_state.logged_in = True
            st.session_state.user_level = user_level
            st.success(f"Masuk Sebagai {user_level} Berhasil!")
        else:
            st.error("Username atau Password Salah!")
else:
    st.sidebar.title("Navigasi")
    if st.session_state.user_level == "admin":
        selected_option = st.sidebar.radio("Pilih Halaman", 
                                           ["Pratinjau Data", "Tambah Data Baru", "Deteksi Sekarang", "Latih Model Naive Bayes", "Langkah-Langkah Perhitungan"])
    else:
        selected_option = st.sidebar.radio("Pilih Halaman", 
                                           ["Deteksi Sekarang"])

    st.title("Deteksi Mandiri Masalah Perangkat Lunak Smartphone Android")

    if st.button("Logout"):
        logout()

    # Upload CSV (hanya untuk admin)
    uploaded_file = st.file_uploader("Silahkan Upload Berkas CSV", type=["csv"]) if st.session_state.user_level == "admin" else None
    df = load_or_create_dataframe(uploaded_file, st.session_state.user_level)

    # Inisialisasi model di session_state jika belum ada
    if 'model' not in st.session_state:
        st.session_state.model = None
        st.session_state.X_test = None

    if selected_option == "Pratinjau Data":
        if st.session_state.user_level == "admin":
            st.write("Pratinjau Data:")
            st.dataframe(df)
        else:
            st.error("Harap Login Kembali")

    elif selected_option == "Tambah Data Baru":
        if st.session_state.user_level == "admin":
            with st.form("Tambah Data"):
                st.write("Tambah Data Baru:")
                new_data = {}
                for col in df.columns:
                    if col != 'Masalah':  # Menyembunyikan kolom target
                        new_data[col] = st.selectbox(f"Enter {col}", options=df[col].unique(), key=f"input_{col}")

                if st.form_submit_button("Simpan Data"):
                    try:
                        # Menyiapkan data
                        features = [col for col in df.columns if col != 'Masalah']
                        X = df[features]
                        y = df['Masalah']
                        
                        # Encode fitur dan target
                        encoders = {col: LabelEncoder().fit(X[col]) for col in X.columns}
                        for col, encoder in encoders.items():
                            X[col] = encoder.transform(X[col])

                        y_encoder = LabelEncoder().fit(y)
                        y = y_encoder.transform(y)

                        # Membuat dan melatih pemodelan Naive Bayes
                        model = CategoricalNB()
                        model.fit(X, y)

                        # Menyiapkan data baru
                        new_data_encoded = {col: encoders[col].transform([new_data[col]])[0] for col in new_data if col != 'Masalah'}

                        # Melakukan prediksi
                        new_data_df = pd.DataFrame([new_data_encoded])
                        new_prediction = model.predict(new_data_df)
                        new_prediction_decoded = y_encoder.inverse_transform(new_prediction)

                        # Menambahkan hasil prediksi ke data baru dan menyimpan ke CSV
                        new_data['Masalah'] = new_prediction_decoded[0]
                        new_data_df = pd.DataFrame([new_data])
                        df = pd.concat([df, new_data_df], ignore_index=True)
                        df.to_csv('DataMurni.CSV', index=False)
                        st.success("Data Berhasil Disimpan!")
                    except Exception as e:
                        st.error(f"Kesalahan Dalam Prediksi: {e}")
        else:
            st.error("Harap Login Kembali")

    elif selected_option == "Deteksi Sekarang":
        new_data = {}
        with st.form("Deteksi Sekarang"):
            
            for col in df.columns:
                if col != 'Masalah':  # Menyembunyikan kolom target
                    new_data[col] = st.selectbox(f"Enter {col}", options=df[col].unique(), key=f"input_{col}_detect")

            if st.form_submit_button("Deteksi"):
                prediction_result = None  # Variable to store prediction result

                try:
                    # Membuat kategorikal features
                    features = [col for col in df.columns if col != 'Masalah']
                    X = df[features]
                    y = df['Masalah']
                    
                    encoders = {col: LabelEncoder().fit(X[col]) for col in X.columns}
                    for col, encoder in encoders.items():
                        X[col] = encoder.transform(X[col])

                    y_encoder = LabelEncoder().fit(y)
                    y = y_encoder.transform(y)

                    # Membuat dan melatih pemodelan Naive Bayes 
                    model = CategoricalNB()
                    model.fit(X, y)

                    # Membuat data baru
                    new_data_encoded = {col: encoders[col].transform([new_data[col]])[0] for col in new_data if col != 'Masalah'}

                    # Membuat prediksi
                    new_data_df = pd.DataFrame([new_data_encoded])
                    new_prediction = model.predict(new_data_df)
                    new_prediction_decoded = y_encoder.inverse_transform(new_prediction)

                    prediction_result = new_prediction_decoded[0]  # Store prediction result

                    # Menampilkan akurasi
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    st.write(f"Akurasi Pemodelan Pada Data Uji: {accuracy * 100:.2f}%")

                except Exception as e:
                    st.error(f"Kesalahan Dalam Prediksi: {e}")

                if prediction_result is not None:
                    if prediction_result == "Berat":
                        st.warning(f"Tingkat Masalah: {prediction_result} - Masalah ini memerlukan penanganan segera, direkomendasikan untuk memperbaikinya kepada ahli.")
                    elif prediction_result == "Ringan":
                        st.info(f"Tingkat Masalah: {prediction_result} - Smartphone anda dapat diperbaiki secara mandiri.")
                        st.info("Berikut langkah-langkah yang dapat dilakukan pada opsi pengaturan pada smartphone anda antara lain:")
                        st.info("- Reset Semua Pengaturan")
                        st.info("- Reset Ke Pengaturan Pabrik")
                    else:
                        st.info(f"Tingkat Masalah: {prediction_result} - Informasi tambahan diperlukan.")

    elif selected_option == "Latih Model Naive Bayes":
        if st.session_state.user_level == "admin":
            # Upload CSV terlebih dahulu
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                st.write("File CSV berhasil diupload. Silakan pilih fitur dan target untuk melatih model.")
            
                with st.form("Latih Model"):
                    st.write("Pilih fitur dan target untuk data latih:")
                    target = st.selectbox("Pilih Target", options=df.columns.tolist(), index=len(df.columns)-1)
                    features = st.multiselect("Pilih Fitur", options=[col for col in df.columns if col != target], default=[col for col in df.columns if col != target])

                    # Slider persentase data uji
                    test_size = st.slider("Pilih Persentase Data Uji (%)", min_value=10, max_value=100, value=20, step=10) / 100

                    if st.form_submit_button("Mulai Melatih"):
                        if features and target:
                            X = df[features]
                            y = df[target]

                            # Encode fitur dan target
                            encoders = {col: LabelEncoder().fit(X[col]) for col in X.columns}
                            for col, encoder in encoders.items():
                                X[col] = encoder.transform(X[col])

                            y_encoder = LabelEncoder().fit(y)
                            y = y_encoder.transform(y)

                            # Split data
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

                            # Latih model
                            model = CategoricalNB()
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)

                            # Simpan model di session_state
                            st.session_state.model = model
                            st.session_state.X_test = X_test
                            st.session_state.y_test = y_test
                            st.session_state.y_pred = y_pred
                            st.session_state.features = features
                            st.session_state.y_encoder = y_encoder
                            
                            st.write("Model telah dilatih.")
                            conf_matrix = confusion_matrix(y_test, y_pred)
                            fig, ax = plt.subplots(figsize=(10, 7))
                            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax)
                            ax.set_xlabel('Prediksi')
                            ax.set_ylabel('Sebenarnya')
                            st.pyplot(fig)

                            # Membaca prediksi kembali ke kategori asli
                            y_test_decoded = y_encoder.inverse_transform(y_test)
                            y_pred_decoded = y_encoder.inverse_transform(y_pred)
                        
                            # Menampilkan hasil prediksi ke data kategorikal
                            X_test_original = df.loc[X_test.index, features]
                            result_df = X_test_original.copy()
                            result_df['Aktual'] = y_test_decoded
                            result_df['Terprediksi'] = y_pred_decoded
                            result_df['Keterangan'] = result_df['Aktual'] == result_df['Terprediksi']
                            st.markdown("### Hasil Prediksi:")
                            st.dataframe(result_df)
                            
                            # Menghitung jumlah prediksi benar dan salah
                            correct_predictions = result_df['Keterangan'].sum()
                            incorrect_predictions = len(result_df) - correct_predictions

                            st.write(f"Jumlah Prediksi Benar: {correct_predictions}")
                            st.write(f"Jumlah Prediksi Salah: {incorrect_predictions}")
                            
                            st.session_state.result_df = result_df
                            
                            st.write("Akurasi Model Naive Bayes:")
                            st.write(f"{accuracy_score(y_test, y_pred) * 100:.2f}%")
                        else:
                            st.error("Harap Pilih Fitur dan Target.")  
                # Tombol unduh di luar formulir
                if 'result_df' in st.session_state:
                    def convert_df_to_csv(df):
                        return df.to_csv(index=False).encode('utf-8')

                    csv = convert_df_to_csv(st.session_state.result_df)

                    st.download_button(
                        label="Unduh Hasil Prediksi sebagai CSV",
                        data=csv,
                        file_name='hasil_prediksi.csv',
                        mime='text/csv',
                    )
    

    elif selected_option == "Langkah-Langkah Perhitungan":
        st.markdown("## Langkah-langkah Perhitungan Naive Bayes")

        # Langkah 1: Menghitung Probabilitas A Priori Kelas P(Ci)
        st.markdown("### Langkah 1: Menghitung Probabilitas A Priori Kelas $P(C_i)$")
        st.markdown("""
        Rumus:
        $$P(C_i) = \\frac{n(C_i)}{n_{total}}$$
        di mana:
        - $P(C_i)$ adalah probabilitas kelas $C_i$
        - $n(C_i)$ adalah jumlah data yang termasuk dalam kelas $C_i$
        - $n_{total}$ adalah jumlah total data
        """)

        if st.session_state.model:
            model = st.session_state.model
            
            # Mengambil Log Probabilitas A Priori Kelas
            class_log_probabilities = model.class_log_prior_
            class_probabilities = np.exp(class_log_probabilities)  # Mengubah dari log probabilitas ke probabilitas biasa
            
            # Hanya tampilkan dua kelas pertama
            num_classes_to_display = min(2, len(class_probabilities))  # Menghindari indeks out of range
            st.write("Probabilitas A Priori Kelas:")
            st.write({f"Klasifikasi {i}": class_probabilities[i] for i in range(num_classes_to_display)})

            # Langkah 2: Menghitung Probabilitas Kondisional Fitur P(Xk|Ci)
            st.markdown("### Langkah 2: Menghitung Probabilitas Kondisional Fitur $P(X_k|C_i)$")
            st.markdown("""
            Rumus:
            $$P(X_k|C_i) = \\frac{n(X_k|C_i) + \\alpha}{n(C_i) + \\alpha \\cdot |X_k|}$$
            di mana:
            - $P(X_k|C_i)$ adalah probabilitas fitur $X_k$ dalam kelas $C_i$
            - $n(X_k|C_i)$ adalah jumlah kemunculan fitur $X_k$ dalam kelas $C_i$
            - $n(C_i)$ adalah jumlah total kemunculan kelas $C_i$
            - $|X_k|$ adalah jumlah total fitur unik
            - $\\alpha$ adalah parameter smoothing (biasanya 1 untuk Laplace smoothing)
            """)
            feature_log_probabilities = model.feature_log_prob_
            st.write("Probabilitas Fitur Bersyarat:")
            for i, feature_log_probs in enumerate(feature_log_probabilities):
                st.write(f"Atribut {i}:")
                st.write({f"Klasifikasi {j}": np.exp(log_prob) for j, log_prob in enumerate(feature_log_probs[:2])})  # Mengubah dari log probabilitas ke probabilitas dan menampilkan hanya 2 fitur

            # Langkah 3: Menghitung Probabilitas Kelas Diberikan Fitur P(Ci|X)
            st.markdown("### Langkah 3: Menghitung Probabilitas Kelas Diberikan Fitur $P(C_i|X)$")
            st.markdown("""
            Rumus:
            $$P(C_i|X) = P(C_i) \\cdot \\prod_{k=1}^{n} P(X_k|C_i)$$
            di mana:
            - $P(C_i|X)$ adalah probabilitas kelas $C_i$ diberikan fitur $X$
            - $P(X_k|C_i)$ adalah probabilitas fitur $X_k$ dalam kelas $C_i$
            - $n$ adalah jumlah total fitur
            """)
            final_log_probabilities = model.predict_log_proba(st.session_state.X_test)
            st.write("Probabilitas Akhir:")
            for i, final_log_probs in enumerate(final_log_probabilities):
                st.write(f"Data Uji {i}:")
                st.write({f"Klasifikasi {j}": np.exp(log_prob) for j, log_prob in enumerate(final_log_probs[:2])})  # Mengubah dari log probabilitas ke probabilitas dan menampilkan hanya 2 kelas

            # Langkah 4: Menentukan Kelas Dengan Probabilitas Tertinggi
            st.markdown("### Langkah 4: Menentukan Kelas Dengan Probabilitas Tertinggi")
            st.markdown("""
            Prediksi kelas $C_i$ untuk data $X$ adalah kelas dengan nilai probabilitas tertinggi:
            $$C_{pred} = \\arg\\max_{C_i} P(C_i|X)$$
            """)
            # Peta kategori
            category_mapping = {0: 'Berat', 1: 'Ringan'}

            # Hasil prediksi numerik
            y_pred_numeric = model.predict(st.session_state.X_test)

            # Konversi hasil prediksi menjadi kategorikal
            y_pred_categorical = [category_mapping[num] for num in y_pred_numeric]

            # Tampilkan hasil prediksi kategorikal
            st.write(f"Hasil prediksi: {y_pred_categorical}")

        else:
            st.error("Model Naive Bayes belum dilatih.")


