import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

# Fungsi untuk mengupload file
def upload_file():
    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            return df
        except Exception as e:
            st.error(f"Error: {e}")
            return None
    else:
        return None

# Fungsi untuk memproses dan memvisualisasikan data
def process_data(data_df):
    try:
        # Mengubah kolom 'Tanggal' menjadi format datetime
        data_df['Tanggal'] = pd.to_datetime(data_df['Tanggal'], format='%d-%b-%y')

        # Mengatur kolom 'Tanggal' sebagai index
        data_df.set_index('Tanggal', inplace=True)

        # Menghapus kolom non-numerik jika ada
        data_df = data_df.select_dtypes(include=[float, int])

        return data_df
    except Exception as e:
        st.error(f"Error dalam memproses data: {e}")
        return None

# Fungsi untuk menampilkan statistik deskriptif
def show_descriptive_statistics(data_df):
    st.subheader("Statistika Deskriptif")
    st.write(data_df.describe())  # Menampilkan statistik deskriptif

# Fungsi untuk plot time series harian
def plot_time_series_daily(data_df: pd.DataFrame, province: str):
    st.subheader(f"Plot Time Series Harian untuk {province}")
    if province in data_df.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(data_df.index, data_df[province], label=province, color='blue')
        ax.set_title(f"Time Series Harian - {province}", fontsize=16)
        ax.set_xlabel('Tanggal', fontsize=12)
        ax.set_ylabel('Nilai', fontsize=12)
        ax.legend(loc='upper left')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

# Fungsi untuk pemetaan dan clustering
def mapping_and_clustering():
    st.title("Pemetaan Clustering dengan DTW")
    
    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

    if uploaded_file is not None:
        data_df = load_and_process_data(uploaded_file)

        # Proses penghitungan dan pemetaan
        # (Berisi kode pemetaan dan penghitungan DTW seperti yang Anda sediakan sebelumnya)

# Fungsi utama aplikasi
def main():
    st.title("Aplikasi Statistika Deskriptif dan Pemetaan")
    
    # Menambahkan sidebar untuk navigasi
    page = st.sidebar.selectbox("Pilih Halaman", ["Statistika Deskriptif", "Pemetaan"])

    if page == "Statistika Deskriptif":
        data_df = upload_file()
        
        if data_df is not None:
            st.subheader("Dataframe")
            st.write(data_df)

            processed_data_df = process_data(data_df)

            if processed_data_df is not None:
                selected_province = st.selectbox("Pilih Provinsi", options=processed_data_df.columns.tolist())
                
                if selected_province:
                    show_descriptive_statistics(processed_data_df[[selected_province]])
                    plot_time_series_daily(processed_data_df, selected_province)

    elif page == "Pemetaan":
        mapping_and_clustering()

if __name__ == "__main__":
    main()
