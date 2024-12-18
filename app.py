import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc, confusion_matrix, silhouette_score
import pickle
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from PIL import Image
import warnings

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Credit Risk Analysis", layout="wide")

# Inisiasi variabel untuk memuat model yang sudah dilatih Gunakan Pickel untuk load model(JANGAN DI HAPUS, ISI DIBAWAH INI)
xgboost = pickle.load(open('xgbmodel.pkl', 'rb'))
kmeans = pickle.load(open('k_means_model.pkl', 'rb'))

# Buat Fungsi yang dapat mengeluarkan hasil prediksi dari model berdasarkan input dari user
def prediction_classification(person_age, person_income, person_home_ownership, person_emp_length, 
               loan_intent, loan_grade, loan_amnt, loan_int_rate, loan_percent_income, 
               cb_person_default_on_file, cb_person_cred_hist_length):

    encoder_columns = ['person_age', 'person_emp_length', 'loan_grade', 'loan_int_rate', 'loan_percent_income',
                       'cb_person_default_on_file', 'cb_person_cred_hist_length', 'loan_to_income_ratio',
                       'loan_to_emp_length_ratio', 'int_rate_to_loan_amt_ratio', 
                       'person_home_ownership_MORTGAGE', 'person_home_ownership_OTHER', 'person_home_ownership_OWN', 
                       'person_home_ownership_RENT', 'loan_intent_DEBTCONSOLIDATION', 'loan_intent_EDUCATION', 
                       'loan_intent_HOMEIMPROVEMENT', 'loan_intent_MEDICAL', 'loan_intent_PERSONAL', 
                       'loan_intent_VENTURE', 'income_group_high', 'income_group_high-middle', 'income_group_low', 
                       'income_group_low-middle', 'income_group_middle', 'loan_amount_group_large', 
                       'loan_amount_group_medium', 'loan_amount_group_small', 'loan_amount_group_very large']
    
    processed_input = preprocess_input_classification(
        person_age, person_income, person_home_ownership, person_emp_length,
        loan_intent, loan_grade, loan_amnt, loan_int_rate, loan_percent_income,
        cb_person_default_on_file, cb_person_cred_hist_length, encoder_columns)

    # Prediksi menggunakan model XGBoost
    pred = xgboost.predict(processed_input)

    return pred

def prediction_clustering(person_age, loan_amnt, loan_percent_income, cb_person_cred_hist_length):
    processed_input = preprocess_input_clustering(person_age, loan_amnt, loan_percent_income, cb_person_cred_hist_length)
    
    # Lakukan prediksi cluster berdasarkan input
    hasil_clustering = kmeans.predict(processed_input)[0]
    
    return hasil_clustering


# Buat fungsi yang dapat mengeluarkan metrik evaluasi model (JANGAN DI HAPUS, ISI DIBAWAH INI)
def evaluate_model_clustering(pca_df):
    sil_score = silhouette_score(pca_df, kmeans.labels_)
    return sil_score

def evaluate_model_classification(X_test, y_test):
    y_pred = xgboost.predict(X_test)
    y_pred_proba = xgboost.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    return accuracy, precision, recall, f1, roc_auc, y_pred, y_pred_proba

# Buat fungsi untuk membuat visualisasi plot kurva ROC (JANGAN DI HAPUS, ISI DIBAWAH INI)
def plot_roc_curve(fpr, tpr, roc_auc):  
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--', lw=2, label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid()
    st.pyplot(plt)

# Buat fungsi untuk membuat visualisasi confusion matrix (JANGAN DI HAPUS, ISI DIBAWAH INI)
def plot_confusion_matrix(cm):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
    plt.xlabel("Predicted Labels")
    plt.ylabel("Actual Labels")
    plt.title("Confusion Matrix")
    st.pyplot(plt)

def plot_elbow(inertia):
    # Plot Elbow
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 11), inertia, marker='o', linestyle='--')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.grid()
    plt.show()
    st.pyplot(plt)

def plot_vizcluster(pca_df):
    # Plot hasil clustering
    centroids = kmeans.cluster_centers_
    plt.figure(figsize=(8, 6))

    # Scatter plot data berdasarkan label cluster
    plt.scatter(pca_df.iloc[:, 0], pca_df.iloc[:, 1], c=pca_df['Cluster'], cmap='viridis', s=50, alpha=0.6)

    # Menambahkan centroid pada plot
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, marker='X', label='Centroids')

    # Menambahkan label dan judul
    plt.title('Clustering Results with KMeans (PCA Components)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()

    # Tampilkan plot
    plt.show()
    st.pyplot(plt)

def prepare_data_classification(inputpath):
    df = pd.read_csv(inputpath)
    # Preprocessing data
    df.drop_duplicates(inplace=True)
    
    # Mengisi nilai yang hilang
    df['loan_int_rate'] = df['loan_int_rate'].fillna(df['loan_int_rate'].mean())
    df['person_emp_length'] = df['person_emp_length'].fillna(df['person_emp_length'].median())

    df = df[df['person_age'] < 100]

    print(df['loan_status'].value_counts())

    # Mendefinisikan fungsi untuk menghapus outlier menggunakan IQR
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    def remove_outliers(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Menghapus baris yang memiliki nilai di luar batas IQR
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    # Terapkan fungsi remove_outliers untuk setiap kolom numerik
    for column in numeric_columns:
        df_cleaned = remove_outliers(df, column)

    # Feature Engineering
    df_cleaned['loan_to_income_ratio'] = df_cleaned['loan_amnt'] / df_cleaned['person_income']
    df_cleaned['loan_to_emp_length_ratio'] = df_cleaned['person_emp_length'] / df_cleaned['loan_amnt']
    df_cleaned['int_rate_to_loan_amt_ratio'] = df_cleaned['loan_int_rate'] / df_cleaned['loan_amnt']

    df_cleaned['income_group'] = pd.cut(df_cleaned['person_income'],
                                        bins=[0, 25000, 50000, 75000, 100000, float('inf')],
                                        labels=['low', 'low-middle', 'middle', 'high-middle', 'high'])
    df_cleaned.drop('person_income', axis=1, inplace=True)

    df_cleaned['loan_amount_group'] = pd.cut(df_cleaned['loan_amnt'],
                                             bins=[0, 5000, 10000, 15000, float('inf')],
                                             labels=['small', 'medium', 'large', 'very large'])
    df_cleaned.drop('loan_amnt', axis=1, inplace=True)

    print(df_cleaned['loan_status'].value_counts())

    # One-Hot Encoding
    columns_to_encode = ['person_home_ownership', 'loan_intent', 'income_group', 'loan_amount_group']
    encoder = OneHotEncoder(sparse_output=False)
    
    for col in columns_to_encode:
        encoded_data = encoder.fit_transform(df_cleaned[[col]])
        encoded_columns = encoder.get_feature_names_out([col])
        encoded_df = pd.DataFrame(encoded_data, columns=encoded_columns, index=df_cleaned.index)
        df_cleaned = pd.concat([df_cleaned.drop(columns=[col]), encoded_df], axis=1)

    # Label Encoding
    columns_to_encode = ['loan_grade', 'cb_person_default_on_file']
    label_encoder = LabelEncoder()
    for col in columns_to_encode:
        df_cleaned[col] = label_encoder.fit_transform(df_cleaned[col])

    # Scaling
    scaler = StandardScaler()
    features_to_scale = df_cleaned.drop('loan_status', axis=1)
    df_scaled = scaler.fit_transform(features_to_scale)
    df_scaled = pd.DataFrame(df_scaled, columns=features_to_scale.columns, index=df_cleaned.index)
    df_scaled['loan_status'] = df_cleaned['loan_status']

    print(df_scaled['loan_status'].value_counts())
    # SMOTE Oversampling
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(df_scaled.drop(columns=['loan_status']), df_scaled['loan_status'])

    # Mengembalikan dataset setelah oversampling
    df_balanced = pd.concat([pd.DataFrame(X_resampled, columns=df_scaled.columns[:-1]), pd.DataFrame(y_resampled, columns=['loan_status'])], axis=1)

    return df_balanced

def prepare_data_clustering(inputpath):
    df = pd.read_csv(inputpath)
    df_selection = df[['person_age', 'loan_amnt', 'loan_percent_income', 'cb_person_cred_hist_length']]

    # Menggunakan StandardScaler untuk normalisasi
    scaler = StandardScaler()
    scaled = scaler.fit(df_selection)
    scaled = scaler.transform(df_selection)
    
    # Reduksi dimensi menggunakan PCA (2 komponen)
    pca = PCA(n_components=2)
    pca_dfa = pca.fit_transform(scaled)
    pca_df = pd.DataFrame(pca_dfa, columns=['PCA1', 'PCA2'])
    pca_df['Cluster'] = kmeans.labels_

    return pca_df

def preprocess_input_classification(person_age, person_income, person_home_ownership, person_emp_length, 
                     loan_intent, loan_grade, loan_amnt, loan_int_rate, loan_percent_income,
                     cb_person_default_on_file, cb_person_cred_hist_length, encoder_columns):

    input_data = pd.DataFrame({
        'person_age': [person_age],
        'person_income': [person_income],
        'person_home_ownership': [person_home_ownership],
        'person_emp_length': [person_emp_length],
        'loan_intent': [loan_intent],
        'loan_grade': [loan_grade],
        'loan_amnt': [loan_amnt],
        'loan_int_rate': [loan_int_rate],
        'loan_percent_income': [loan_percent_income],
        'cb_person_default_on_file': [cb_person_default_on_file],
        'cb_person_cred_hist_length': [cb_person_cred_hist_length]
    })

    print(input_data.head())
    data_encoding2 = pd.read_csv("data_encoding2.csv")

    # Feature Engineering
    input_data['loan_to_income_ratio'] = input_data['loan_amnt'] / input_data['person_income']
    input_data['loan_to_emp_length_ratio'] = input_data['person_emp_length'] / input_data['loan_amnt']
    input_data['int_rate_to_loan_amt_ratio'] = input_data['loan_int_rate'] / input_data['loan_amnt']

    # Income Grouping
    input_data['income_group'] = pd.cut(input_data['person_income'],
                                         bins=[0, 25000, 50000, 75000, 100000, float('inf')],
                                         labels=['low', 'low-middle', 'middle', 'high-middle', 'high'])
    input_data.drop('person_income', axis=1, inplace=True)

    # Loan Amount Grouping
    input_data['loan_amount_group'] = pd.cut(input_data['loan_amnt'],
                                              bins=[0, 5000, 10000, 15000, float('inf')],
                                              labels=['small', 'medium', 'large', 'very large'])
    input_data.drop('loan_amnt', axis=1, inplace=True)

    # One-Hot Encoding: Menggunakan handle_unknown='ignore' agar tidak error jika kategori baru
    categorical_cols = ['person_home_ownership', 'loan_intent', 'income_group', 'loan_amount_group']
    encoder = OneHotEncoder(sparse_output=False)
    encoded_data = encoder.fit_transform(input_data[categorical_cols])
    encoded_columns = encoder.get_feature_names_out(categorical_cols)
    encoded_df = pd.DataFrame(encoded_data, columns=encoded_columns)

    # Drop kolom asli setelah encoding
    input_data = input_data.drop(columns=categorical_cols).reset_index(drop=True)

    # Gabungkan hasil encoding dengan data numerik
    input_data = pd.concat([input_data, encoded_df], axis=1)

    # Label Encoding untuk loan_grade dan cb_person_default_on_file
    label_encoder = LabelEncoder()
    input_data['loan_grade'] = label_encoder.fit_transform(input_data['loan_grade'])
    input_data['cb_person_default_on_file'] = label_encoder.fit_transform(input_data['cb_person_default_on_file'])

    # Pastikan semua kolom yang diharapkan oleh model ada di input
    all_columns = encoder_columns  # Kolom yang diharapkan dari model
    missing_columns = set(all_columns) - set(input_data.columns)

    # Tambahkan kolom yang hilang dengan nilai 0
    for col in missing_columns:
        input_data[col] = 0

    # Susun kolom dengan urutan yang sesuai dengan model
    input_data = input_data[all_columns]

    # Scaling data
    scaler = StandardScaler()
    scaled_data = scaler.fit(data_encoding2)
    scaled_data = scaler.transform(input_data)
    input_data_scaled = pd.DataFrame(scaled_data, columns=input_data.columns)

    return input_data_scaled

def preprocess_input_clustering(person_age, loan_amnt, loan_percent_income, cb_person_cred_hist_length):
    df = pd.read_csv('data_outlier1.csv')
    df_selection = df[['person_age', 'loan_amnt', 'loan_percent_income', 'cb_person_cred_hist_length']]
    # Membuat DataFrame dari input yang diterima
    input_data = pd.DataFrame([[person_age, loan_amnt, loan_percent_income, cb_person_cred_hist_length]], 
                              columns=['person_age', 'loan_amnt', 'loan_percent_income', 'cb_person_cred_hist_length'])

    # Menggunakan StandardScaler untuk normalisasi
    scaler = StandardScaler()
    input_scaled = scaler.fit(df_selection)
    input_scaled_train = scaler.transform(df_selection)
    input_scaled_input = scaler.transform(input_data)
    
    # Reduksi dimensi menggunakan PCA (2 komponen)
    pca = PCA(n_components=2)
    pca_transformed = pca.fit(input_scaled_train)
    pca_transformed_train = pca.transform(input_scaled_train)
    pca_transformed_input = pca.transform(input_scaled_input)
    
    # Mengembalikan DataFrame dengan dua komponen PCA
    pca_df_train = pd.DataFrame(pca_transformed_train, columns=['PCA1', 'PCA2'])
    pca_df_input = pd.DataFrame(pca_transformed_input, columns=['PCA1', 'PCA2'])

    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans.fit(pca_df_train)

    return pca_df_input

# --- NAVIGATION MENU ---
def main():
    # Horizontal navigation menu
    menu_selection = option_menu(
        menu_title=None,
        options=["Dashboard (Overview)", "Business Understanding", "Data Understanding", 
                 "Exploratory Data Analysis", "Data Preprocessing", "Modeling & Evaluation", 
                 "Input Credit Risk Detection"],
        icons=["house-fill", "info-circle", "bar-chart-line", 
               "binoculars", "wrench-adjustable", "robot", "file-earmark-text"],
        default_index=0,
        orientation="horizontal",
    )


    # --- PAGE 1: HOME ---
    if menu_selection == "Dashboard (Overview)":
        st.markdown("""
            <style>
            /* Custom CSS for the Home Page */
            .title-text {
                text-align: center;
                font-size: 40px;
                font-weight: bold;
                color: #FF6F61;
                text-shadow: 1px 1px 2px #000000;
                margin-top: 30px;
                margin-bottom: 20px;
            }
            .subtitle-text {
                text-align: center;
                font-size: 24px;
                color: #555555;
                margin-top: -15px;
                margin-bottom: 40px;
            }
            .content-box {
                padding: 20px;
                background-color: #FFFFFF;
                border-radius: 10px;
                box-shadow: 0px 0px 10px rgba(0,0,0,0.2);
            }
            .table-container {
                margin: auto;
                width: 70%;
                text-align: center;
                border-collapse: collapse;
            }
            .table-container th, .table-container td {
                border: 1px solid #dddddd;
                padding: 10px;
                font-size: 16px;
            }
            .table-container th {
                background-color: #FF6F61;
                color: white;
                font-weight: bold;
            }
            .table-container td {
                background-color: #ffffff;
                color: black;
            }
            </style>
            
            <div class="title-text">CREDIT RISK DETECTION WEBSITE</div>
            <div class="subtitle-text">"Analisis Risiko Kredit untuk Prediksi dan Segmentasi Pelanggan Berdasarkan Karakteristik Kredit"</div>
            <br>
            
            <div class="content-box">
                <p style="font-size: 18px; text-align: justify; color: black; text-align: center;">
                    Proyek ini bertujuan untuk melakukan <b>analisis risiko kredit</b> menggunakan teknik <b>klasifikasi</b> dan <b>clustering</b>.
                    Dengan memanfaatkan model machine learning, kami mengklasifikasikan pelanggan ke dalam kategori risiko kredit 
                    yang aman atau berisiko serta melakukan segmentasi pelanggan berdasarkan karakteristik kredit mereka.
                </p>
                <p style="font-size: 18px; text-align: justify; color: black; text-align: center;">
                    Pendekatan ini menggunakan kerangka kerja <b>CRISP-DM</b> yang terdiri dari tahapan <i>Business Understanding</i>, 
                    <i>Data Understanding</i>, <i>Data Preparation</i>, <i>Modeling</i>, dan <i>Evaluation</i>. 
                    Dengan metode ini, analisis dapat dilakukan secara sistematis dan akurat.
                </p>
            </div>
            
            <br>
            <br>
            <h3 style="text-align:center; color: #FF6F61; font-weight: bold;">ANGGOTA KELOMPOK</h3>
            <br>
            <table class="table-container">
                <tr>
                    <th>Nama Anggota</th>
                    <th>NIM</th>
                </tr>
                <tr>
                    <td>Ahmad Fauzi</td>
                    <td>1202220263</td>
                </tr>
                <tr>
                    <td>Maryam Grischelda Ardety Wijaya</td>
                    <td>1202223262</td>
                </tr>
                <tr>
                    <td>Matthew Alexander Hasintongan Sitorus</td>
                    <td>1202223168</td>
                </tr>
                <tr>
                    <td>Muhammad Addin Zulfikar</td>
                    <td>1202220132</td>
                </tr>
            </table>
        """, unsafe_allow_html=True)

        
    # --- PAGE 2: BUSINESS UNDERSTANDING ---
    elif menu_selection == "Business Understanding":
        # Custom CSS
        st.markdown("""
            <style>
            /* General Styling */
            .title {
                text-align: center;
                font-size: 38px;
                font-weight: bold;
                color: #FF6F61;
                margin-bottom: 20px;
            }
            .subtitle {
                text-align: center;
                font-size: 24px;
                color: #555555;
                margin-top: -15px;
                margin-bottom: 40px;
            }
            .section-header {
                font-size: 22px;
                font-weight: bold;
                color: #FF6F61;
                margin-top: 20px;
                margin-bottom: 10px;
            }
            .info-box {
                background-color: #F9F9F9;
                color: black;
                padding: 15px;
                border-left: 7px solid #FF6F61;
                border-radius: 5px;
                margin-bottom: 20px;
                box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.1);
            }
            .highlight {
                font-weight: bold;
                color: #FF6F61;
            }
            </style>
        """, unsafe_allow_html=True)

        # Title
        st.markdown('<div class="title">💼 BUSINESS UNDERSTANDING 💼</div>', unsafe_allow_html=True)
        st.markdown('<div class="subtitle">Analisis Risiko Kredit dalam Industri Keuangan</div>', unsafe_allow_html=True)
        st.markdown("---", unsafe_allow_html=True)
        st.markdown("""
            <style>
            .image-container {
                display: flex;
                justify-content: center;
                margin-bottom: 20px;
            }
            .rounded-image {
                border-radius: 15px;  /* Atur nilai border-radius sesuai kebutuhan */
                width: 60%;           /* Atur lebar gambar */
            }
            </style>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="image-container"><img class="rounded-image" src="https://cdn.hswstatic.com/gif/debt-management-credit-card.jpg"></div>', unsafe_allow_html=True)

        # Section: Introduction to Credit Risk
        st.markdown('<div class="section-header">🔍 Risiko Kredit</div>', unsafe_allow_html=True)
        st.markdown("""
            Risiko kredit merupakan tantangan utama dalam industri keuangan yang dapat memengaruhi stabilitas dan profitabilitas lembaga keuangan, seperti bank dan perusahaan pembiayaan. 
            Risiko ini terjadi ketika seorang peminjam gagal memenuhi kewajibannya untuk membayar pinjaman sesuai jadwal, yang sering kali diukur melalui tingkat kredit macet atau **Non-Performing Loan (NPL)**.
            Tingginya tingkat NPL dapat mengurangi kepercayaan masyarakat terhadap stabilitas sistem keuangan.
        """, unsafe_allow_html=True)

        st.markdown("""
            <div class="info-box">
                Risiko kredit dapat mengarah pada kerugian finansial yang besar bagi lembaga keuangan. 
                Oleh karena itu, penting untuk memodelkan risiko ini guna meningkatkan keputusan pemberian kredit dan pengelolaan portofolio yang lebih efektif.
            </div>
        """, unsafe_allow_html=True)

        # Section: Project Objective
        st.markdown('<div class="section-header">🎯 Tujuan Proyek</div>', unsafe_allow_html=True)
        st.markdown("""
            Tujuan utama dari proyek ini adalah untuk membangun dua jenis model machine learning, yaitu **Classification** dan **Clustering**, menggunakan **XGBoost Classifier** dan **K-means** untuk menganalisis dataset risiko kredit.
            Dataset ini mencakup fitur-fitur utama seperti usia, pendapatan, status pekerjaan, tujuan pinjaman, riwayat kredit, dan status pembayaran.
        """, unsafe_allow_html=True)

        st.markdown("""
            <div class="info-box">
                Dengan model ini, diharapkan dapat meminimalkan kerugian akibat gagal bayar dengan memberikan prediksi yang lebih akurat tentang kelayakan pinjaman.
            </div>
        """, unsafe_allow_html=True)

        # Section: Key Aspects of the Project
        st.markdown('<div class="section-header">🔑 Aspek Penting Proyek</div>', unsafe_allow_html=True)
        st.markdown("""
            1. **Memahami Pola Risiko:** Mengidentifikasi pola yang mengindikasikan potensi risiko gagal bayar berdasarkan data historis nasabah.
            2. **Optimisasi Manajemen Risiko:** Dengan model yang lebih akurat, perusahaan dapat meminimalkan kerugian dan mengoptimalkan portofolio kredit.
            3. **Pendukung Keputusan:** Model yang dikembangkan diharapkan dapat menjadi alat pendukung keputusan yang meningkatkan akurasi dalam penilaian risiko kredit.
        """, unsafe_allow_html=True)

        st.markdown("""
            <div class="info-box">
                Pemahaman yang lebih baik tentang risiko kredit memungkinkan lembaga keuangan untuk mengelola eksposur terhadap risiko yang lebih baik.
            </div>
        """, unsafe_allow_html=True)

        # Section: Benefits for Organization and Stakeholders
        st.markdown('<div class="section-header">💡 Manfaat untuk Organisasi dan Stakeholder</div>', unsafe_allow_html=True)
        st.markdown("""
            1. **Classification (XGBoost):** Model ini membantu organisasi dalam memperkuat pengambilan keputusan terkait persetujuan pinjaman dengan memprediksi kemungkinan pelunasan dan mengurangi risiko kerugian akibat gagal bayar.
            2. **Clustering (K-Means):** Model ini membantu dalam segmentasi nasabah, memungkinkan organisasi untuk menyesuaikan strategi pemasaran dan kebijakan produk sesuai dengan kebutuhan setiap segmen nasabah.
        """, unsafe_allow_html=True)

        # Section: Model Success Criteria
        st.markdown('<div class="section-header">🏆 Nilai Keberhasilan Model</div>', unsafe_allow_html=True)
        st.markdown("""
            Keberhasilan proyek ini akan dinilai berdasarkan:
            - **Classification (XGBoost):** Akurasi yang tinggi pada model dalam memprediksi risiko gagal bayar nasabah.
            - **Clustering (K-Means):** Keberhasilan dalam mengelompokkan nasabah dengan pola serupa untuk analisis lebih lanjut.
        """, unsafe_allow_html=True)

    # --- PAGE 3: DATA UNDERSTANDING ---
    elif menu_selection == "Data Understanding":
        df = pd.read_csv('credit_risk_dataset.csv')
        # Custom CSS
        st.markdown("""
            <style>
            /* General Styling */
            .title {
                text-align: center;
                font-size: 38px;
                font-weight: bold;
                color: #FF6F61;
                margin-bottom: 20px;
            }
            .subtitle {
                text-align: center;
                font-size: 24px;
                color: #555555;
                margin-top: -15px;
                margin-bottom: 40px;
            }
            .section-header {
                font-size: 22px;
                font-weight: bold;
                color: #FF6F61;
                margin-top: 20px;
                margin-bottom: 10px;
            }
            .info-box {
                background-color: #F9F9F9;
                color: black;
                padding: 15px;
                border-left: 7px solid #FF6F61;
                border-radius: 5px;
                margin-bottom: 20px;
                box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.1);
            }
            .highlight {
                font-weight: bold;
                color: #FF6F61;
            }
            .dataframe-container {
                margin-top: 10px;
                margin-bottom: 20px;
                text-align: center;
            }
            </style>
        """, unsafe_allow_html=True)

        # Title
        st.markdown('<div class="title">📊 DATA UNDERSTANDING 📊</div>', unsafe_allow_html=True)
        st.markdown('<div class="subtitle">Eksplorasi Awal Dataset Risiko Kredit</div>', unsafe_allow_html=True)
        st.markdown("---", unsafe_allow_html=True)

        # Dataset Preview
        st.markdown('<div class="section-header">🔍 Dataset Preview</div>', unsafe_allow_html=True)
        st.write("Berikut adalah preview dari dataset yang digunakan dalam analisis risiko kredit:")
        st.dataframe(df.head(), use_container_width=True)

        # Shape of Dataset
        st.markdown('<div class="section-header">📝 Shape of the Dataset</div>', unsafe_allow_html=True)
        st.markdown(f"""
            <div class="info-box">
                Dataset memiliki <span class="highlight">{df.shape[0]}</span> baris dan 
                <span class="highlight">{df.shape[1]}</span> kolom.
            </div>
        """, unsafe_allow_html=True)

        # Data Types
        st.markdown('<div class="section-header">📋 Data Types</div>', unsafe_allow_html=True)
        st.markdown(f"""
            <div class="info-box">
                Dataset ini terdiri dari dua jenis kolom:
                <ul>
                    <li>Kolom Kategorikal: <span class="highlight">{len(df.select_dtypes(include=['object']).columns)}</span> kolom</li>
                    <li>Kolom Numerik: <span class="highlight">{len(df.select_dtypes(include=['int64', 'float64']).columns)}</span> kolom</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

        # Missing Values
        st.markdown('<div class="section-header">⚠️ Missing Values</div>', unsafe_allow_html=True)
        st.write("Menghitung jumlah nilai yang hilang (null) untuk setiap kolom:")

        missing_values = df.isnull().sum()
        missing_values = missing_values[missing_values > 0].sort_values(ascending=False)

        missing_df = pd.DataFrame({
            "Column Name": missing_values.index,
            "Missing Values Count": missing_values.values
        })

        st.dataframe(missing_df, use_container_width=True)

        # Duplicate Data
        st.markdown('<div class="section-header">🔄 Duplicated Data</div>', unsafe_allow_html=True)
        duplicated_rows = df.duplicated().sum()
        st.markdown(f"""
            <div class="info-box">
                Jumlah baris yang memiliki data duplikat: <span class="highlight">{duplicated_rows}</span>
            </div>
        """, unsafe_allow_html=True)

        # Deskripsi Statistik
        st.markdown('<div class="section-header">📈 Deskripsi Statistik Kolom Numerik</div>', unsafe_allow_html=True)
        st.dataframe(df.describe(), use_container_width=True)

        st.markdown('<div class="section-header">📊 Deskripsi Statistik Kolom Kategorikal</div>', unsafe_allow_html=True)
        st.dataframe(df.describe(include='object'), use_container_width=True)

    elif menu_selection == "Exploratory Data Analysis":
        df = pd.read_csv('credit_risk_dataset.csv')

        # Custom CSS
        st.markdown("""
            <style>
            /* General Styling */
            .title {
                text-align: center;
                font-size: 38px;
                font-weight: bold;
                color: #FF6F61;
                margin-bottom: 20px;
            }
            .subtitle {
                text-align: center;
                font-size: 24px;
                color: #555555;
                margin-top: -15px;
                margin-bottom: 40px;
            }
            .section-header {
                font-size: 22px;
                font-weight: bold;
                color: #FF6F61;
                margin-top: 20px;
                margin-bottom: 10px;
            }
            </style>
        """, unsafe_allow_html=True)

        # Title
        st.markdown('<div class="title">🔎 DATA EXPLORATION 🔎</div>', unsafe_allow_html=True)
        st.markdown('<div class="subtitle">Eksplorasi Data Risiko Kredit dengan Visualisasi</div>', unsafe_allow_html=True)
        st.markdown("---", unsafe_allow_html=True)

        # --- Korelasi Matriks ---
        numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
        st.markdown('<div class="section-header">🔗 Korelasi Antar Variabel</div>', unsafe_allow_html=True)
        plt.figure(figsize=(12, 8))
        correlation_matrix = df[numerical_features].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix', fontsize=16)
        st.pyplot(plt.gcf())  # Menampilkan di Streamlit

        # --- Pairplot ---
        # Display a section header
        st.markdown('<div class="section-header">🔍 Hubungan Antar Variabel</div>', unsafe_allow_html=True)
        st.write("Pairplot menunjukkan hubungan antar variabel numerik berdasarkan status pinjaman:")

        # Load and display the image
        image = Image.open('outputpairplot.png')
        st.image(image, caption="Hubungan Antar Variabel", use_column_width=True)

        # --- Boxplot ---
        st.markdown('<div class="section-header">📦 Distribusi Boxplot Fitur Numerik</div>', unsafe_allow_html=True)
        st.write("Boxplot membantu memahami distribusi dan outlier dari fitur numerik:")

        num_plots = len(numerical_features)
        cols = 2  # 2 kolom
        rows = (num_plots + 1) // cols

        fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(18, 5 * rows))
        axes = axes.flatten()

        for i, col in enumerate(numerical_features):
            sns.boxplot(x=df[col], ax=axes[i], color='skyblue')
            axes[i].set_title(f'Boxplot of {col}', fontsize=14)
            axes[i].set_xlabel('')
            axes[i].set_ylabel('')

        for j in range(i + 1, len(axes)):  # Matikan axis yang tidak digunakan
            fig.delaxes(axes[j])

        plt.tight_layout()
        st.pyplot(fig)

        # --- Visualisasi 3x2 Grid ---
        st.markdown('<div class="section-header">📊 Distribusi dan Perbandingan Fitur</div>', unsafe_allow_html=True)

        # List fitur yang ingin divisualisasikan
        features = [
            ('loan_status', 'Distribusi Status Pinjaman'),
            ('person_age', 'Distribusi Status Pinjaman Berdasarkan Usia'),
            ('person_home_ownership', 'Status Kepemilikan Rumah Berdasarkan Loan Status'),
            ('loan_grade', 'Distribusi Loan Grade'),
            ('loan_intent', 'Distribusi Loan Intent'),
            ('cb_person_default_on_file', 'Status Gagal Bayar Historis')
        ]

        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(18, 18))
        axes = axes.flatten()

        for i, (feature, title) in enumerate(features):
            sns.countplot(x=feature, hue='loan_status', data=df,
                        palette={0: 'lightgreen', 1: 'lightcoral'}, ax=axes[i])
            axes[i].set_title(title, fontsize=14)
            axes[i].set_xlabel('')
            axes[i].set_ylabel('Count')

            # Kondisi khusus untuk person_age
            if feature in ['loan_intent', 'person_age']:
                for label in axes[i].get_xticklabels():
                    label.set_fontsize(8)
                axes[i].tick_params(axis='x', rotation=90)

            # Ubah legenda
            handles, labels = axes[i].get_legend_handles_labels()
            axes[i].legend(handles, ['Good Loans', 'Bad Loans'], title='Loan Status')

        plt.tight_layout()
        st.pyplot(fig)


    # --- PAGE 4: DATA PREPARATION ---
    elif menu_selection == "Data Preprocessing":
        tab1, tab2 = st.tabs(["Classification Preparation 📝", "Clustering Preparation 📝"])
        df = pd.read_csv('credit_risk_dataset.csv')
        # Custom CSS
        st.markdown("""
            <style>
            /* General Styling */
            .title {
                text-align: center;
                font-size: 38px;
                font-weight: bold;
                color: #FF6F61;
                margin-bottom: 20px;
            }
            .subtitle {
                text-align: center;
                font-size: 24px;
                color: #555555;
                margin-top: -15px;
                margin-bottom: 40px;
            }
            .section-header {
                font-size: 25px;
                font-weight: bold;
                color: #FF6F61;
                margin-top: 20px;
                margin-bottom: 10px;
            }
            .section-header2 {
                font-size: 20px;
                font-weight: bold;
                color: #FF6F61;
                margin-top: 20px;
                margin-bottom: 10px;
            }
            .info-box {
                background-color: #F9F9F9;
                color: black;
                padding: 15px;
                border-left: 7px solid #FF6F61;
                border-radius: 5px;
                margin-bottom: 20px;
                box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.1);
            }
            .highlight {
                font-weight: bold;
                color: #FF6F61;
            }
            .dataframe-container {
                margin-top: 10px;
                margin-bottom: 20px;
                text-align: center;
            }
            </style>
        """, unsafe_allow_html=True)

        with tab1:
            st.write("<br>", unsafe_allow_html=True)
            st.markdown('<div class="title">⚙️ DATA PREPROCESSING ⚙️</div>', unsafe_allow_html=True)
            st.markdown('<div class="subtitle">Persiapan Data untuk Model Klasifikasi</div>', unsafe_allow_html=True)

            st.markdown("---", unsafe_allow_html=True)

            st.markdown('<div class="section-header">🧹 Data Cleaning</div>', unsafe_allow_html=True)
            # Null Value Handling
            st.markdown('<div class="section-header2">1. Null Value Handling ⚠️</div>', unsafe_allow_html=True)

            # Sebelum Handling
            st.write("Jumlah nilai null sebelum dilakukan penanganan:")
            missing_values_before = df.isnull().sum()
            missing_df_before = pd.DataFrame({
                "Column Name": missing_values_before.index,
                "Missing Values Count": missing_values_before.values
            }).query("`Missing Values Count` > 0")
            st.dataframe(missing_df_before, use_container_width=True)

            st.markdown(f"""
                <div class="info-box">
                    Untuk menangani nilai null pada kolom, berikut adalah langkah-langkah yang dilakukan:
                    <ul>
                        <li>Isi nilai null (imputasi) pada kolom <span class="highlight">'loan_int_rate'</span> dengan nilai rata-rata (mean).</li>
                        <li>Isi nilai null (imputasi) pada kolom <span class="highlight">'person_emp_length'</span> dengan nilai median.</li>
                    </ul>
                    Berikut adalah kode yang digunakan:
                    <br>
                    <br>
                    <pre><code>
            df['loan_int_rate'] = df['loan_int_rate'].fillna(df['loan_int_rate'].mean())<br>
            df['person_emp_length'] = df['person_emp_length'].fillna(df['person_emp_length'].median())
                    </code></pre>
                </div>
            """, unsafe_allow_html=True)


            # Handling Missing Values
            df['loan_int_rate'] = df['loan_int_rate'].fillna(df['loan_int_rate'].mean())
            df['person_emp_length'] = df['person_emp_length'].fillna(df['person_emp_length'].median())
            st.write("Jumlah nilai null setelah dilakukan penanganan:")
            missing_values_after = df.isnull().sum()
            missing_df_after = pd.DataFrame({
                "Column Name": missing_values_after.index,
                "Missing Values Count": missing_values_after.values
            }).query("`Missing Values Count` > 0")
            st.dataframe(missing_df_after, use_container_width=True)

            # Duplicated Data Handling
            st.markdown('<div class="section-header2">2. Duplicated Data Handling 🔄</div>', unsafe_allow_html=True)
            duplicated_before = df.duplicated().sum()
            df = df.drop_duplicates()
            duplicated_after = df.duplicated().sum()
            st.markdown(f"""
                <div class="info-box">
                    Jumlah data duplikat sebelum penanganan: <span class="highlight">{duplicated_before}</span><br>
                    Jumlah data duplikat setelah penanganan: <span class="highlight">{duplicated_after}</span>
                    <br>
                    <br>
                    <pre><code>
                    df.drop_duplicates(inplace=True)
                    </code></pre>
                </div>
                </div>
            """, unsafe_allow_html=True)

            # Outlier Handling
            df_outliers = pd.read_csv("data_outlier.csv")
            st.markdown('<div class="section-header2">3. Outlier Handling 💀</div>', unsafe_allow_html=True)
            numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
            before_outliers = df.shape[0]
            after_outliers = df_outliers.shape[0]
            st.markdown(f"""
                <div class="info-box">
                    Menghapus data outlier menggunakan metode IQR:<br>
                    <ul>
                        <li>Jumlah baris sebelum outlier handling: <span class="highlight">{before_outliers}</span></li>
                        <li>Jumlah baris setelah outlier handling: <span class="highlight">{after_outliers}</span></li>
                    </ul>
                    <br>
                    <pre>

                    def remove_outliers(df, column):
                        Q1 = df[column].quantile(0.25)
                        Q3 = df[column].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

                    for col in numerical_features:
                        data_outlier = remove_outliers(df, col)

                </div>
            """, unsafe_allow_html=True)
            
            # Boxplots
            st.write("Visualisasi Setelah Outlier Handling (Boxplots):")
            cols = 4
            num_plots = len(numerical_features)
            rows = (num_plots + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
            axes = axes.flatten()
            for i, col in enumerate(numerical_features):
                if i < num_plots:
                    sns.boxplot(x=df[col], ax=axes[i], color='skyblue')
                    axes[i].set_title(f"Boxplot of {col}")
            for i in range(num_plots, len(axes)):
                fig.delaxes(axes[i])
            plt.tight_layout()
            st.pyplot(fig)


            st.markdown(f"""
                <br>
                <div class="info-box">
                    Setelah diidentifikasi lebih lanjut, ternyata pada fitur person_age terdapat data yang tidak masuk akal (anomali).
                    <span class="highlight">Terdapat umur yang bernilai lebih dari 100 tahun</span>, yang kemungkinan merupakan data yang tidak valid. Oleh karena itu,
                    kita perlu melakukan penanganan lebih lanjut terhadap data tersebut. Menggunakan:
                    <br><br>
                    <pre>

                    data_outlier = data_outlier[data_outlier['person_age'] < 100]

                </div>
            """, unsafe_allow_html=True)

            st.markdown("---", unsafe_allow_html=True)

            df_outliers2 = pd.read_csv("data_outlier1.csv")
            # Feature Engineering
            st.markdown('<div class="section-header">🔧 Feature Engineering</div>', unsafe_allow_html=True)

            st.markdown(f"""
                <div class="info-box">
                    Feature Engineering adalah proses menambahkan fitur baru untuk meningkatkan kinerja model. Fitur-fitur yang ditambahkan:
                    <ul>
                        <li><strong>loan_to_income_ratio</strong>: Rasio jumlah pinjaman terhadap pendapatan pribadi.</li>
                        <li><strong>loan_to_emp_length_ratio</strong>: Rasio lama pekerjaan terhadap jumlah pinjaman.</li>
                        <li><strong>int_rate_to_loan_amt_ratio</strong>: Rasio suku bunga terhadap jumlah pinjaman.</li>
                    </ul><br>
                    <pre>

                    data['loan_to_income_ratio'] = data['loan_amnt'] / data['person_income']

                    data['loan_to_emp_length_ratio'] =  data['person_emp_length']/ data['loan_amnt'] 

                    data['int_rate_to_loan_amt_ratio'] = data['loan_int_rate'] / data['loan_amnt']
                </div>
            """, unsafe_allow_html=True)


            # Menambahkan fitur-fitur baru
            df_outliers2['loan_to_income_ratio'] = df_outliers2['loan_amnt'] / df_outliers2['person_income']
            df_outliers2['loan_to_emp_length_ratio'] = df_outliers2['person_emp_length'] / df_outliers2['loan_amnt']
            df_outliers2['int_rate_to_loan_amt_ratio'] = df_outliers2['loan_int_rate'] / df_outliers2['loan_amnt']

            # Menampilkan dataframe dengan fitur baru
            st.dataframe(df_outliers2.head(), use_container_width=True)

            st.markdown("---", unsafe_allow_html=True)
            
            # Data Transformation
            st.markdown('<div class="section-header">🔄 Data Transformation</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-header2">1. Discretization (Data Binning) 🗑️</div>', unsafe_allow_html=True)

            st.markdown(f"""
                <div class="info-box">
                    <strong>Discretization (Data Binning)</strong> adalah teknik untuk mengubah variabel numerik menjadi kategori yang lebih sederhana dengan membagi rentang nilai menjadi beberapa interval atau "bin". Teknik ini berguna untuk:
                    <ul>
                        <li><strong>Income Group</strong>: Mengelompokkan data pendapatan pribadi ke dalam beberapa kategori berdasarkan rentang nilai yang telah ditentukan (misalnya: low, low-middle, middle, high-middle, dan high).</li>
                        <li><strong>Loan Amount Group</strong>: Mengelompokkan jumlah pinjaman menjadi beberapa kategori, seperti small, medium, large, dan very large, untuk mempermudah analisis dan interpretasi data.</li>
                    </ul><br>
                    <pre>
                        
                    # Income Group
                    data['income_group'] = pd.cut(data['person_income'],
                                                bins=[0, 25000, 50000, 75000, 100000, float('inf')],
                                                labels=['low', 'low-middle', 'middle', 'high-middle', 'high'])
                    data.drop('person_income', axis=1, inplace=True)

                    # Loan Amount Group
                    data['loan_amount_group'] = pd.cut(data['loan_amnt'],
                                                    bins=[0, 5000, 10000, 15000, float('inf')],
                                                    labels=['small', 'medium', 'large', 'very large'])
                    data.drop('loan_amnt', axis=1, inplace=True)
                </div>
            """, unsafe_allow_html=True)

            df_outliers2['income_group'] = pd.cut(df_outliers2['person_income'], bins=[0, 25000, 50000, 75000, 100000, float('inf')],
                                                labels=['low', 'low-middle', 'middle', 'high-middle', 'high'])
            df_outliers2['loan_amount_group'] = pd.cut(df_outliers2['loan_amnt'], bins=[0, 5000, 10000, 15000, float('inf')],
                                                    labels=['small', 'medium', 'large', 'very large'])
            st.dataframe(df_outliers2[['income_group', 'loan_amount_group']], use_container_width=True)

            # Encoding
            st.markdown('<div class="section-header2">2. Data Encoding 🏷️</div>', unsafe_allow_html=True)
            st.markdown(f"""
                <div class="info-box">
                    <strong>One-Hot Encoding</strong> adalah teknik untuk mengubah variabel kategorikal menjadi variabel numerik dengan cara membuat kolom baru untuk setiap kategori dan memberi nilai 1 jika baris memiliki kategori tersebut, dan 0 jika tidak. Teknik ini digunakan untuk variabel kategorikal yang tidak memiliki urutan, misalnya:
                    <ul>
                        <li><strong>'person_home_ownership' (Home Ownership):</strong> Kolom ini menunjukkan apakah individu memiliki rumah atau tidak. Nilai-nilai kategori seperti 'own', 'mortgage', dan 'rent' adalah nominal, yaitu tidak ada urutan atau ranking di antara kategori tersebut. Oleh karena itu, <strong>One-Hot Encoding</strong> dipilih karena tidak ada hubungan urutan yang perlu dipertimbangkan.</li>
                        <li><strong>'loan_intent' (Loan Intent):</strong> Kolom ini menunjukkan tujuan dari pinjaman yang diajukan oleh individu. Kategori seperti 'personal', 'education', 'medical', 'venture', 'debt consolidation', dll juga merupakan kategori nominal. Seperti variabel sebelumnya, <strong>One-Hot Encoding</strong> adalah teknik yang tepat untuk menghindari urutan yang tidak ada antara kategori.</li>
                        <li><strong>'income_group' (Income Group):</strong> Ini adalah kolom yang dibuat untuk mengelompokkan data pendapatan individu ke dalam beberapa kategori, seperti 'low', 'low-middle', 'middle', 'high-middle', dan 'high'. Meskipun ini adalah variabel kategorikal, <strong>One-Hot Encoding</strong> dipilih untuk menjaga interpretasi yang lebih sederhana dan memungkinkan model untuk menangani grup sebagai kategori independen.</li>
                        <li><strong>'loan_amount_group' (Loan Amount Group):</strong> Kolom ini mengelompokkan jumlah pinjaman ke dalam kategori seperti 'small', 'medium', 'large', dan 'very large'. Ini adalah variabel kategorikal nominal, jadi <strong>One-Hot Encoding</strong> akan digunakan untuk menghindari menganggap adanya urutan atau ranking di antara kategori tersebut.</li>
                    </ul>
                    <pre>
                        
                    from sklearn.preprocessing import OneHotEncoder

                    # Membuat fungsi untuk melakukan one-hot encoding
                    def one_hot_encoding(data, columns):    
                        encoder = OneHotEncoder(sparse_output=False)
                        
                        for col in columns:
                            # Fit dan transform untuk setiap fitur
                            encoded_data = encoder.fit_transform(data[[col]])  # Hasilnya adalah array 2D

                            # Ambil nama fitur yang benar untuk setiap kategori
                            encoded_columns = encoder.get_feature_names_out([col])  # Nama untuk setiap fitur

                            # Membuat DataFrame baru dengan hasil encoding
                            encoded_df = pd.DataFrame(encoded_data, columns=encoded_columns, index=data.index)

                            # Gabungkan hasil encoding dengan data
                            data = pd.concat([data.drop(columns=[col]), encoded_df], axis=1)

                        return data


                    # Contoh penggunaan untuk mengencode kolom
                    columns_to_encode = ['person_home_ownership', 'loan_intent', 'income_group', 'loan_amount_group']

                    # Lakukan one hot encoding pada semua kolom ini
                    data = one_hot_encoding(data, columns_to_encode)
                </div>

                <div class="info-box">
                    <strong>Label Encoding</strong> adalah teknik untuk mengubah variabel kategorikal menjadi angka dengan cara memberi label numerik pada setiap kategori. Teknik ini digunakan pada variabel kategorikal yang memiliki urutan atau ranking, misalnya:
                    <ul>
                        <li><strong>'loan_grade' (Loan Grade):</strong> Kolom ini menunjukkan kategori kualitas pinjaman berdasarkan penilaian kredit. Nilai seperti 'A', 'B', 'C', dan seterusnya memiliki urutan tertentu, yang berarti 'A' lebih baik dari 'B', dan seterusnya. Oleh karena itu, <strong>Label Encoding</strong> dipilih untuk mengubah kategori ini menjadi angka untuk menjaga urutan antar kategori tersebut.</li>
                        <li><strong>'cb_person_default_on_file' (Default on File):</strong> Ini adalah kolom yang menunjukkan apakah seseorang memiliki riwayat gagal bayar atau default pada file kredit mereka. Nilainya adalah 'yes' dan 'no', yang meskipun bersifat biner, bisa dianggap sebagai variabel ordinal, karena 'yes' menunjukkan default yang lebih buruk daripada 'no'. Oleh karena itu, <strong>Label Encoding</strong> akan mengkodekan nilai-nilai ini sebagai angka (0 dan 1) dengan mempertahankan makna ordinal tersebut.</li>
                    </ul>
                    <pre>
                        
                    from sklearn.preprocessing import LabelEncoder

                    # Membuat fungsi untuk melakukan label encoding
                    def label_encoding(data, columns):
                        encoder = LabelEncoder()

                        for col in columns:
                            data[col] = encoder.fit_transform(data[col])
                        
                        return data


                    # Fitur yang akan dikodekan
                    columns_to_encode = ['loan_grade', 'cb_person_default_on_file']

                    # Lakukan label encoding pada semua kolom ini
                    data = label_encoding(data, columns_to_encode)
                </div>
            """, unsafe_allow_html=True)

            data = pd.read_csv("data_encoding.csv")
            encoded_columns = [
                'loan_grade', 'cb_person_default_on_file', 'person_home_ownership_MORTGAGE', 'person_home_ownership_OTHER', 'person_home_ownership_OWN', 'person_home_ownership_RENT',
                'loan_intent_DEBTCONSOLIDATION', 'loan_intent_EDUCATION', 'loan_intent_HOMEIMPROVEMENT', 'loan_intent_MEDICAL',
                'loan_intent_PERSONAL', 'loan_intent_VENTURE', 'income_group_high', 'income_group_high-middle', 'income_group_low',
                'income_group_low-middle', 'income_group_middle', 'loan_amount_group_large', 'loan_amount_group_medium',
                'loan_amount_group_small', 'loan_amount_group_very large'
            ]
            st.dataframe(data[encoded_columns], use_container_width=True)

            # Scaling
            st.markdown('<div class="section-header2">3. Data Scaling 📏</div>', unsafe_allow_html=True)
            st.markdown(f"""
                <div class="info-box">
                    Data scaling atau normalisasi adalah proses mengubah nilai fitur numerik menjadi skala yang lebih seragam. Hal ini berguna karena:
                    <ul>
                        <li><strong>StandardScaler</strong> mengubah data dengan menghitung nilai rata-rata dan deviasi standar dari setiap fitur, sehingga setiap fitur memiliki distribusi dengan mean = 0 dan std = 1.</li>
                        <li>Scaling ini mencegah fitur dengan rentang nilai yang lebih besar mendominasi model, serta meningkatkan kinerja model dalam beberapa algoritma seperti KNN, SVM, atau gradient-based algorithms.</li>
                    </ul>
                    Berikut adalah kode yang digunakan untuk scaling data:
                    <br><br>
                    <pre>
                        
                    from sklearn.preprocessing import StandardScaler

                    # Membuat objek scaler
                    scaler = StandardScaler()

                    # Menyaring fitur yang akan discaling (kecuali 'loan_status')
                    features_to_scale = data.drop('loan_status', axis=1) 

                    # Melakukan transformasi scaling
                    data_scaled = scaler.fit_transform(features_to_scale) 

                    # Mengubah hasil scaling menjadi DataFrame dengan nama kolom yang sama
                    data_scaled = pd.DataFrame(data_scaled, columns=features_to_scale.columns, index=data.index)

                    # Menambahkan kolom 'loan_status' kembali
                    data_scaled['loan_status'] = data['loan_status']
                </div>
            """, unsafe_allow_html=True)

            data_scaled = pd.read_csv("data_scaled.csv")
            st.dataframe(data_scaled, use_container_width=True)

            # Handling Imbalanced Target
            st.markdown('<div class="section-header2">4. Handling Imbalanced Target ⚖️</div>', unsafe_allow_html=True)

            st.markdown(f"""
            <div class="info-box">
                <strong>SMOTE (Synthetic Minority Over-sampling Technique)</strong> adalah teknik untuk menangani masalah <strong>imbalanced data</strong>, di mana kelas target memiliki distribusi yang tidak seimbang (misalnya, banyak sampel untuk satu kelas dan sedikit untuk kelas lainnya). SMOTE bekerja dengan:
                <ul>
                    <li>Menambahkan sampel sintetis untuk kelas minoritas dengan memanipulasi data yang ada, bukan hanya dengan duplikasi data.</li>
                    <li>Setiap sampel minoritas yang ada akan memiliki sampel baru yang dibuat berdasarkan jarak terdekat dengan sampel yang sudah ada.</li>
                </ul>
                Dengan menggunakan SMOTE, kita dapat meningkatkan kinerja model dengan membuat distribusi target lebih seimbang, sehingga model dapat belajar lebih baik untuk mengklasifikasikan kedua kelas.
            </div>
            <div class="info-box">
                <strong>Distribusi Target Data Sebelum dan Setelah SMOTE:</strong><br>
                <table style="width:100%; border-collapse: collapse;">
                    <tr style="border-bottom: 2px solid #ddd;">
                        <th style="padding: 8px; border: 1px solid #ddd; text-align: left;">Condition</th>
                        <th style="padding: 8px; border: 1px solid #ddd; text-align: left;">Class 0</th>
                        <th style="padding: 8px; border: 1px solid #ddd; text-align: left;">Class 1</th>
                    </tr>
                    <tr style="border-bottom: 1px solid #ddd;">
                        <td style="padding: 8px; border: 1px solid #ddd;">Before SMOTE</td>
                        <td style="padding: 8px; border: 1px solid #ddd;">24431 baris</td>
                        <td style="padding: 8px; border: 1px solid #ddd;">6842 baris</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; border: 1px solid #ddd;">After SMOTE</td>
                        <td style="padding: 8px; border: 1px solid #ddd;">24431 baris</td>
                        <td style="padding: 8px; border: 1px solid #ddd;">24431 baris</td>
                    </tr>
                </table>
            </div>
            """, unsafe_allow_html=True)
        
        with tab2:
            st.write("<br>", unsafe_allow_html=True)
            st.markdown('<div class="title">⚙️ DATA PREPARATION ⚙️</div>', unsafe_allow_html=True)
            st.markdown('<div class="subtitle">Persiapan Data untuk Model Klustering</div>', unsafe_allow_html=True)

            st.markdown("---", unsafe_allow_html=True)

            # Feature Selection
            st.markdown('<div class="section-header">🔍 Feature Selection for Clustering Model</div>', unsafe_allow_html=True)

            st.markdown("""
                <br>
                <div class="info-box">
                    <strong>Feature Selection</strong> adalah langkah penting dalam persiapan data sebelum membangun model clustering. Dalam proses ini, hanya fitur yang relevan dan memiliki kontribusi signifikan terhadap clustering yang dipilih, sementara fitur yang tidak relevan atau redundant akan dihapus. Ini membantu meningkatkan efisiensi dan kualitas model.
                    <br><br>
                    Fitur yang dipilih untuk model clustering adalah:
                    <ul>
                        <li><strong>person_age</strong>: Umur individu yang dapat mempengaruhi kemampuan atau perilaku finansial.</li>
                        <li><strong>loan_amnt</strong>: Jumlah pinjaman yang diminta, yang bisa menjadi indikator kebutuhan finansial.</li>
                        <li><strong>loan_percent_income</strong>: Persentase pendapatan yang digunakan untuk pinjaman, menunjukkan seberapa besar beban pinjaman terhadap pendapatan individu.</li>
                        <li><strong>cb_person_cred_hist_length</strong>: Durasi riwayat kredit, yang memberikan gambaran mengenai kestabilan keuangan individu.</li>
                    </ul>
                    <br>
                    Mengapa memilih fitur ini?
                    <ul>
                        <li><strong>Relevansi:</strong> Fitur-fitur ini berhubungan langsung dengan analisis finansial dan perilaku individu, yang relevan untuk membedakan kelompok dalam clustering.</li>
                        <li><strong>Pengurangan Dimensi:</strong> Mengurangi jumlah fitur yang digunakan dapat meningkatkan kinerja model dan mempercepat proses komputasi.</li>
                        <li><strong>Efisiensi:</strong> Memilih fitur yang paling informatif menghindari overfitting dan memastikan model lebih mudah untuk menggeneralisasi ke data baru.</li>
                    </ul>
                    <br>
                    <pre><code>
            data_clustering = data_outlier[['person_age', 'loan_amnt', 'loan_percent_income', 'cb_person_cred_hist_length']]
                    </code></pre>
                </div>
            """, unsafe_allow_html=True)

            st.markdown("---", unsafe_allow_html=True)

            # Data Scaling
            st.markdown('<div class="section-header">📏 Data Scaling for Clustering Model</div>', unsafe_allow_html=True)

            st.markdown("""
                <br>
                <div class="info-box">
                    <strong>Data Scaling</strong> adalah proses mengubah skala fitur sehingga mereka berada pada skala yang sama, yang sangat penting dalam model clustering. Clustering, seperti K-Means, mengandalkan jarak antar data untuk menentukan kelompok. Oleh karena itu, fitur dengan skala yang lebih besar dapat mendominasi jarak dan mempengaruhi hasil clustering secara tidak proporsional. 
                    <br><br>
                    Mengapa kita melakukan scaling pada data?
                    <ul>
                        <li><strong>Menjaga Keadilan Antar Fitur:</strong> Beberapa fitur mungkin memiliki skala yang jauh lebih besar (misalnya, pendapatan atau jumlah pinjaman) dibandingkan dengan fitur lain (misalnya, umur). Tanpa scaling, fitur yang memiliki skala lebih besar bisa mendominasi clustering.</li>
                        <li><strong>Mempercepat Konvergensi Model:</strong> Dalam model berbasis jarak seperti K-Means, proses konvergensi bisa lebih cepat ketika data berada dalam skala yang seragam.</li>
                        <li><strong>Menjaga Konsistensi:</strong> Scaling membantu memastikan bahwa tidak ada fitur yang lebih dominan dari yang lainnya, sehingga model bisa memanfaatkan seluruh data secara optimal.</li>
                    </ul>
                    <br>
                    <pre><code>
            scaler = StandardScaler()
            data_clustering_scaled = scaler.fit_transform(data_clustering)
            data_clustering_scaled = pd.DataFrame(data_clustering_scaled, columns=data_clustering.columns)
                    </code></pre>
                </div>
            """, unsafe_allow_html=True)

            data_clustering_scaled = pd.read_csv("data_clustering_scaled.csv")
            st.dataframe(data_clustering_scaled, use_container_width=True)

            st.markdown("---", unsafe_allow_html=True)

            # PCA (Principal Component Analysis)
            st.markdown('<div class="section-header">🔍 Principal Component Analysis (PCA)</div>', unsafe_allow_html=True)

            st.markdown("""
                <br>
                <div class="info-box">
                    <strong>Principal Component Analysis (PCA)</strong> adalah teknik untuk mengurangi dimensi data sambil mempertahankan sebanyak mungkin informasi yang ada. PCA sangat berguna dalam clustering karena sering kali data memiliki banyak fitur yang saling berkorelasi, yang bisa membuat model lebih kompleks dan sulit untuk dianalisis.
                    <br><br>
                    <strong>Kenapa menggunakan PCA?</strong>
                    <ul>
                        <li><strong>Mengurangi Dimensi:</strong> PCA mengurangi jumlah fitur yang digunakan dalam model tanpa kehilangan informasi yang penting, membuat model menjadi lebih efisien dan mudah untuk dianalisis.</li>
                        <li><strong>Meningkatkan Interpretabilitas:</strong> Dengan mengurangi dimensi, PCA memungkinkan kita untuk melihat data dalam dua atau tiga dimensi, yang lebih mudah untuk dianalisis dan divisualisasikan.</li>
                        <li><strong>Mengurangi Kolinearitas:</strong> PCA membantu mengurangi korelasi antar fitur, yang dapat meningkatkan kualitas hasil clustering.</li>
                    </ul>
                    <br>
                    <pre>
                        
                    from sklearn.decomposition import PCA

                    pca = PCA(n_components=2) 
                    df_pca = pca.fit_transform(data_clustering_scaled)

                    pca_df = pd.DataFrame(df_pca, columns=['PCA1', 'PCA2'])
                    pca_df.head()
                </div>
            """, unsafe_allow_html=True)

            pca_df = pd.read_csv("pca_result.csv")
            st.dataframe(pca_df, use_container_width=True)

    # --- PAGE 6: MODELING & EVALUATION ---
    elif menu_selection == "Modeling & Evaluation":
        st.write("<br>", unsafe_allow_html=True)
        html_temp = """
        <div style="background-color:#FF6F61;padding:13px; border-radius:15px; margin-bottom:20px;">
        <h1 style="color:white;text-align:center;">Modelling & Evaluation</h1>
        </div>
        """
        st.markdown(html_temp, unsafe_allow_html=True)

        tab1, tab2 = st.tabs(["Classification Modelling & Evaluation 🤖", "Clustering Modelling & Evaluation 🎈"])

        with tab1:
            # Memanggil fungsi prepare_data untuk memproses data
            df_balanced = prepare_data_classification("credit_risk_dataset.csv")

            # Membagi data
            X = df_balanced.drop(columns=['loan_status'])
            y = df_balanced['loan_status']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Evaluasi model
            accuracy, precision, recall, f1, roc_auc, y_pred, y_pred_proba = evaluate_model_classification(X_test, y_test)

            # Menampilkan metrik evaluasi model
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.success(f"Accuracy: {accuracy:.2f}")
            with col2:
                st.info(f"Precision: {precision:.2f}")
            with col3:
                st.warning(f"Recall: {recall:.2f}")
            with col4:
                st.error(f"F1 Score: {f1:.2f}")

            # Pilihan visualisasi
            plot_option = st.selectbox("Select the plot to display:", ["Select", "ROC AUC Curve", "Confusion Matrix"])

            if plot_option == "ROC AUC Curve":
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                plot_roc_curve(fpr, tpr, roc_auc)

            elif plot_option == "Confusion Matrix":
                cm = confusion_matrix(y_test, y_pred)
                plot_confusion_matrix(cm)

        with tab2:
            df_clustering = prepare_data_clustering("data_outlier1.csv")
            sil_score = evaluate_model_clustering(df_clustering)
            st.markdown(
                f"""
                <div style="background-color: #06402b; padding: 20px; border-radius: 10px; text-align: center;">
                    <h4 style="color: white;">Silhouette Score: {sil_score:.2f}</h4>
                </div>
                """, unsafe_allow_html=True)
            
            # Pilihan visualisasi
            plot_option = st.selectbox("Select the plot to display:", ["Select", "Elbow Method Curve", "Visualize Clusters"])

            if plot_option == "Elbow Method Curve":
                inertia = []
                for k in range(1,11):
                    kmeans = KMeans(n_clusters=k, random_state=42)
                    kmeans.fit(df_clustering)
                    inertia.append(kmeans.inertia_)
                plot_elbow(inertia)

            elif plot_option == "Visualize Clusters":
                plot_vizcluster(df_clustering)

    # --- PAGE 7: INPUT TEST DATA ---
    elif menu_selection == "Input Credit Risk Detection":
        st.write("<br>", unsafe_allow_html=True)
        html_temp = """
        <div style="background-color:#FF6F61;padding:13px; border-radius:15px; margin-bottom:20px;">
        <h1 style="color:white;text-align:center;">Input Credit Risk Detection</h1>
        </div>
        """
        st.markdown(html_temp, unsafe_allow_html=True)
        st.write("<br>", unsafe_allow_html=True)

        with st.form("prediction_form"):
            # Input fitur
            person_age = st.number_input("Person Age", min_value=18, max_value=100, step=1)
            person_income = st.number_input("Person Income", min_value=0, step=1000)
            person_home_ownership = st.selectbox("Person Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
            person_emp_length = st.number_input("Person Employment Length (years)", min_value=0.0, step=0.5)
            loan_intent = st.selectbox("Loan Intent", ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
            loan_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
            loan_amnt = st.number_input("Loan Amount", min_value=0, step=100)
            loan_int_rate = st.number_input("Loan Interest Rate (%)", min_value=0.0, step=0.1)
            loan_percent_income = st.number_input("Loan Percent of Income", min_value=0.0, max_value=1.0, step=0.01)
            cb_person_default_on_file = st.selectbox("Default on File", ["Yes", "No"])
            cb_person_cred_hist_length = st.number_input("Credit History Length (years)", min_value=0, step=1)

            # Submit button
            st.write("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("Predict", use_container_width=True)
        
        result = ""
        if submitted:
            # Prediksi hasil
            result = prediction_classification(
                person_age, person_income, person_home_ownership, person_emp_length, 
                loan_intent, loan_grade, loan_amnt, loan_int_rate, loan_percent_income, 
                cb_person_default_on_file, cb_person_cred_hist_length
            )

            result = 'Nasabah Berpotensi Gagal Bayar' if result[0] == 1 else 'Nasabah Berpotensi Tidak Gagal Bayar'

            hasil_clustering = prediction_clustering(person_age, loan_amnt, loan_percent_income, cb_person_cred_hist_length)

            # Deskripsi hasil cluster
            cluster_descriptions = {
                0: "Nasabah dengan potensi kredit tinggi, kemungkinan besar mampu memenuhi kewajiban pinjaman.",
                1: "Nasabah dengan potensi kredit moderat, evaluasi lebih lanjut diperlukan.",
                2: "Nasabah dengan potensi kredit rendah, perlu waspada terhadap kemungkinan risiko gagal bayar.",
                3: "Nasabah dengan potensi risiko gagal bayar tinggi, perlu dilakukan mitigasi atau penundaan persetujuan."
            }
            
            # Mendapatkan deskripsi berdasarkan cluster label
            resultclust = cluster_descriptions.get(hasil_clustering, "Cluster tidak dikenali.")

            # Custom CSS
            st.markdown("""
                <style>
                    .card {
                        background-color: #f8f9fa;
                        border-radius: 15px;
                        padding: 20px;
                        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                        margin: 20px auto;
                        width: 100%;
                        max-width: 1000px; /* Full width but max at 1000px */
                        text-align: center;
                        color: #333;
                    }
                    
                    .card h3 {
                        font-size: 24px;
                        font-weight: bold;
                        color: #f0d500;
                    }
                    
                    .warning {
                        background-color: #f32013;
                        color: #F0D500;
                        padding: 15px;
                        border-radius: 8px;
                        font-weight: bold;
                    }
                    
                    .success {
                        background-color: #06402b;
                        color: #388e3c;
                        padding: 15px;
                        border-radius: 8px;
                        font-weight: bold;
                    }

                    .emoji {
                        font-size: 28px;
                    }

                    /* Styling paragraph to make text white */
                    .card p {
                        color: white;
                        font-size: 18px;
                        margin-top: 10px;
                    }
                </style>
            """, unsafe_allow_html=True)

            # Prediksi klasifikasi
            if result == 'Nasabah Berpotensi Gagal Bayar':
                classification_card = f"""
                    <div class="card warning">
                        <div class="emoji">⚠️</div>
                        <h3><strong>Prediksi Classification: {result}</strong></h3>
                        <p><em>Nasabah ini menunjukkan potensi risiko gagal bayar. Disarankan untuk melakukan 
                        evaluasi lebih mendalam dan mengambil langkah-langkah mitigasi, seperti menawarkan 
                        restrukturisasi pinjaman atau menghubungi nasabah untuk klarifikasi lebih lanjut.</em></p>
                    </div>
                """
            else:
                classification_card = f"""
                    <div class="card success">
                        <div class="emoji">✅</div>
                        <h3><strong>Prediksi Classification: {result}</strong></h3>
                        <p><em>Nasabah ini diprediksi tidak memiliki risiko gagal bayar, sehingga proses persetujuan 
                        pinjaman dapat dilanjutkan. Pastikan semua dokumentasi dan persyaratan lainnya sudah lengkap 
                        sebelum diproses lebih lanjut.</em></p>
                    </div>
                """

            if hasil_clustering == 0:
                cluster_card = f"""
                    <div class="card success">
                        <div class="emoji">🌟</div>
                        <h3><strong>Prediksi Clustering: {resultclust}</strong></h3>
                        <p><em>Nasabah ini berada dalam cluster dengan potensi kredit tinggi, dapat melanjutkan proses pinjaman.</em></p>
                    </div>
                """
            elif hasil_clustering == 1:
                cluster_card = f"""
                    <div class="card warning">
                        <div class="emoji">⚠️</div>
                        <h3><strong>Prediksi Clustering: {resultclust}</strong></h3>
                        <p><em>Nasabah ini berada dalam cluster dengan potensi kredit moderat, perlu evaluasi lebih lanjut.</em></p>
                    </div>
                """
            elif hasil_clustering == 2:
                cluster_card = f"""
                    <div class="card warning">
                        <div class="emoji">⚠️</div>
                        <h3><strong>Prediksi Clustering: {resultclust}</strong></h3>
                        <p><em>Nasabah ini berada dalam cluster dengan potensi kredit rendah, risiko gagal bayar perlu diperhatikan.</em></p>
                    </div>
                """
            else:  # hasil_clustering == 3
                cluster_card = f"""
                    <div class="card warning">
                        <div class="emoji">⚠️</div>
                        <h3><strong>Prediksi Clustering: {resultclust}</strong></h3>
                        <p><em>Nasabah ini berada dalam cluster dengan potensi risiko gagal bayar tinggi, langkah mitigasi diperlukan.</em></p>
                    </div>
                """

            # Tampilkan kedua hasil (klasifikasi dan clustering) bersamaan
            st.markdown(classification_card, unsafe_allow_html=True)
            st.markdown(cluster_card, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
