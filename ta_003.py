import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
import numpy as np

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

# Halaman Statistika Deskriptif
def statistik_deskriptif():
    st.subheader("Statistika Deskriptif")
    data_df = upload_file()

    if data_df is not None:
        st.write("Dataframe:")
        st.write(data_df)
        
        # Mengubah kolom 'Tanggal' menjadi format datetime dan mengatur sebagai index
        data_df['Tanggal'] = pd.to_datetime(data_df['Tanggal'], format='%d-%b-%y')
        data_df.set_index('Tanggal', inplace=True)

        # Dropdown untuk memilih provinsi
        selected_province = st.selectbox("Pilih Provinsi", options=data_df.columns.tolist())
        
        if selected_province:
            # Menampilkan statistik deskriptif
            st.subheader(f"Statistika Deskriptif untuk {selected_province}")
            st.write(data_df[selected_province].describe())

            # Menampilkan plot line chart
            st.subheader(f"Line Chart untuk {selected_province}")
            plt.figure(figsize=(12, 6))
            plt.plot(data_df.index, data_df[selected_province], label=selected_province, color='blue')
            plt.title(f"Line Chart - {selected_province}")
            plt.xlabel('Tanggal')
            plt.ylabel('Nilai')
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(plt)

# Halaman Pemetaan
def pemetaan():
    st.subheader("Pemetaan Clustering dengan DTW")
    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"], key="pemetaan_upload")

    if uploaded_file is not None:
        data_df = pd.read_csv(uploaded_file)
        data_df['Tanggal'] = pd.to_datetime(data_df['Tanggal'], format='%d-%b-%y')
        data_df.set_index('Tanggal', inplace=True)
        
        # Proses clustering...
        # (isi sesuai dengan kode pemetaan yang Anda miliki)

# Fungsi utama aplikasi
def main():
    st.title("Aplikasi Statistika Deskriptif dan Pemetaan")
    page = st.sidebar.radio("Pilih Halaman", ("Statistika Deskriptif", "Pemetaan"))

    if page == "Statistika Deskriptif":
        statistik_deskriptif()
    elif page == "Pemetaan":
        pemetaan()

if __name__ == "__main__":
    main()
