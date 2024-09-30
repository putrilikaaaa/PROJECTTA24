import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import squareform
import requests

# URL GeoJSON
URL_GEOJSON = "https://raw.githubusercontent.com/putrilikaaaa/PROJECTTA24/main/indonesia-prov.geojson"

# Fungsi untuk mengupload file CSV
def upload_csv_file(key):
    uploaded_file = st.file_uploader("Upload file CSV", type="csv", key=key)
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return None

# Fungsi untuk pemetaan
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
        
        # Periksa nama kolom yang ada di GeoDataFrame
        st.write("Nama kolom dalam GeoDataFrame:", gdf.columns.tolist())

        # Menghitung kluster dari hasil klustering
        clustered_data = pd.DataFrame({
            'Province': data_daily_standardized.columns,
            'Cluster': labels  # Use the labels from clustering
        })

        # Normalisasi nama provinsi
        clustered_data['Province'] = clustered_data['Province'].str.upper().str.replace('.', '', regex=False).str.strip()

        # Ganti nama provinsi yang tidak konsisten
        if 'Province' in gdf.columns:
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
        else:
            st.error("Kolom 'Province' tidak ditemukan dalam GeoDataFrame.")

# Fungsi utama
def main():
    st.title("Aplikasi Clustering dengan DTW")
    pemetaan()

if __name__ == "__main__":
    main()
