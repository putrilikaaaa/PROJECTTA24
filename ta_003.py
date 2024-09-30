import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
import geopandas as gpd
import urllib.request

# URL GeoJSON dari GitHub
URL_GEOJSON = 'https://raw.githubusercontent.com/putrilikaaaa/PROJECTTA24/main/indonesia-prov.geojson'

# Fungsi untuk mengupload file CSV
def upload_csv_file(key=None):
    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"], key=key)
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            return df
        except Exception as e:
            st.error(f"Error: {e}")
    return None

# Halaman Statistika Deskriptif
def statistik_deskriptif():
    st.subheader("Statistika Deskriptif")
    data_df = upload_csv_file()

    if data_df is not None:
        st.write("Dataframe:")
        st.write(data_df)
        
        # Mengubah kolom 'Tanggal' menjadi format datetime dan mengatur sebagai index
        data_df['Tanggal'] = pd.to_datetime(data_df['Tanggal'], format='%d-%b-%y', errors='coerce')
        data_df.set_index('Tanggal', inplace=True)

        if data_df.isnull().any().any():
            st.warning("Terdapat tanggal yang tidak valid, silakan periksa data Anda.")

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

# Fungsi untuk menghitung matriks biaya lokal DTW
def compute_local_cost_matrix(data_df: pd.DataFrame) -> np.array:
    num_provinces = data_df.shape[1]
    num_time_points = data_df.shape[0]
    local_cost_matrix = np.zeros((num_time_points, num_provinces, num_provinces))

    for i in range(num_provinces):
        for j in range(num_provinces):
            if i != j:
                cost = np.square(data_df.iloc[:, i] - data_df.iloc[:, j])
                local_cost_matrix[:, i, j] = cost

    return local_cost_matrix

# Fungsi untuk menghitung matriks biaya akumulatif
def compute_accumulated_cost_matrix(local_cost_matrix: np.array) -> np.array:
    num_time_points = local_cost_matrix.shape[0]
    num_provinces = local_cost_matrix.shape[1]
    accumulated_cost_matrix = np.zeros((num_time_points, num_provinces, num_provinces))

    for i in range(num_provinces):
        for j in range(num_provinces):
            accumulated_cost_matrix[:, i, j] = np.cumsum(local_cost_matrix[:, i, j])

    return accumulated_cost_matrix

# Fungsi untuk menghitung matriks jarak DTW
def compute_dtw_distance_matrix(accumulated_cost_matrix: np.array) -> np.array:
    num_provinces = accumulated_cost_matrix.shape[1]
    dtw_distance_matrix = np.zeros((num_provinces, num_provinces))

    for i in range(num_provinces):
        for j in range(num_provinces):
            dtw_distance_matrix[i, j] = accumulated_cost_matrix[-1, i, j]

    return dtw_distance_matrix

# Halaman Pemetaan
def pemetaan():
    st.subheader("Pemetaan Clustering dengan DTW")
    data_df = upload_csv_file(key="pemetaan_upload")

    if data_df is not None:
        data_df['Tanggal'] = pd.to_datetime(data_df['Tanggal'], format='%d-%b-%y', errors='coerce')
        data_df.set_index('Tanggal', inplace=True)

        # Menghitung rata-rata harian
        data_daily = data_df.resample('D').mean()

        # Standarisasi data harian
        scaler = StandardScaler()
        data_daily_standardized = pd.DataFrame(scaler.fit_transform(data_daily), index=data_daily.index, columns=data_daily.columns)

        # Hitung matriks biaya lokal dan akumulatif untuk data harian
        local_cost_matrix_daily = compute_local_cost_matrix(data_daily_standardized)
        accumulated_cost_matrix_daily = compute_accumulated_cost_matrix(local_cost_matrix_daily)

        # Hitung matriks jarak DTW untuk data harian
        dtw_distance_matrix_daily = compute_dtw_distance_matrix(accumulated_cost_matrix_daily)

        # Klustering dan perhitungan skor siluet untuk data harian
        max_n_clusters = 10
        silhouette_scores = {}
        best_n_clusters = 2

        for n_clusters in range(2, max_n_clusters + 1):
            clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='complete')
            labels = clustering.fit_predict(dtw_distance_matrix_daily)
            score = silhouette_score(dtw_distance_matrix_daily, labels, metric='precomputed')
            silhouette_scores[n_clusters] = score
            if score > silhouette_scores.get(best_n_clusters, -1):
                best_n_clusters = n_clusters

        # Plot Silhouette Scores
        plt.figure(figsize=(10, 6))
        plt.plot(list(silhouette_scores.keys()), list(silhouette_scores.values()), marker='o', linestyle='-')
        plt.title('Silhouette Score vs. Number of Clusters (Data Harian)')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')
        plt.xticks(range(2, max_n_clusters + 1))
        plt.grid(True)
        st.pyplot(plt)

        # Klustering dan dendrogram
        clustering = AgglomerativeClustering(n_clusters=best_n_clusters, metric='precomputed', linkage='complete')
        labels = clustering.fit_predict(dtw_distance_matrix_daily)
        
        condensed_dtw_distance_matrix = squareform(dtw_distance_matrix_daily)
        Z = linkage(condensed_dtw_distance_matrix, method='complete')

        plt.figure(figsize=(16, 10))
        dendrogram(Z, labels=data_daily_standardized.columns, leaf_rotation=90)
        plt.title('Dendrogram Clustering dengan DTW (Data Harian)')
        plt.xlabel('Provinsi')
        plt.ylabel('Jarak DTW')
        st.pyplot(plt)

        # Memuat GeoDataFrame dari URL GeoJSON
        gdf = gpd.read_file(URL_GEOJSON)

        # Menghitung kluster dari hasil klustering
        clustered_data = pd.DataFrame({
            'Province': data_daily_standardized.columns,
            'Cluster': labels  # Use the labels from clustering
        })

        # Normalisasi nama provinsi
        clustered_data['Province'] = clustered_data['Province'].str.upper().str.replace('.', '', regex=False).str.strip()
        
        # Mengganti nama provinsi yang tidak konsisten
        gdf['Province'] = gdf['Province'].replace({
            'DI ACEH': 'ACEH',
            'KEPULAUAN BANGKA BELITUNG': 'BANGKA BELITUNG',
            'NUSATENGGARA BARAT': 'NUSA TENGGARA BARAT',
            'D.I YOGYAKARTA': 'DI YOGYAKARTA',
            'DAERAH ISTIMEWA YOGYAKARTA': 'DI YOGYAKARTA',
        })

        # Menghapus provinsi yang None (yaitu GORONTALO)
        gdf = gdf[gdf['Province'].notna()]

        # Menggabungkan data terkluster dengan GeoDataFrame
        gdf = gdf.merge(clustered_data, on='Province', how='left')

        # Set warna untuk kluster
        gdf['color'] = gdf['Cluster'].map({0: 'green', 1: 'yellow', 2: 'red'})
        gdf['color'].fillna('grey', inplace=True)

        # Menampilkan tabel kluster
        st.subheader("Tabel Kluster Provinsi:")
        cluster_table = clustered_data.groupby('Cluster')['Province'].apply(list).reset_index()
        cluster_table.columns = ['Cluster', 'Provinces']
        st.write(cluster_table)

        # Plot peta
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        gdf.boundary.plot(ax=ax, linewidth=1, color='black')  # Plot batas
        gdf.plot(ax=ax, color=gdf['color'], edgecolor='black', alpha=0.6)  # Plot provinsi dengan warna

        # Tambahkan judul dan label
        plt.title("Pemetaan Kluster Provinsi Berdasarkan DTW")
        plt.axis('off')
        st.pyplot(fig)

# Fungsi utama
def main():
    st.title("Aplikasi Clustering dan Pemetaan")
    menu = ["Statistika Deskriptif", "Pemetaan"]
    choice = st.sidebar.selectbox("Pilih Halaman", menu)

    if choice == "Statistika Deskriptif":
        statistik_deskriptif()
    elif choice == "Pemetaan":
        pemetaan()

if __name__ == "__main__":
    main()
