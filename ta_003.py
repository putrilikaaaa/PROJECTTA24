import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn_extra.cluster import KMedoids
import geopandas as gpd
import requests
import json

# Fungsi untuk mengunggah file GeoJSON dari GitHub
def upload_geojson_file():
    url = "https://raw.githubusercontent.com/username/repo/main/indonesia-prov.geojson"
    response = requests.get(url)
    data = response.json()
    return gpd.GeoDataFrame.from_features(data['features'])

# Fungsi untuk menghitung matriks biaya lokal DTW
def compute_local_cost_matrix(data):
    # Implementasi fungsi DTW
    pass

# Fungsi untuk menghitung matriks biaya akumulatif
def compute_accumulated_cost_matrix(local_cost_matrix):
    # Implementasi fungsi untuk menghitung matriks biaya akumulatif
    pass

# Fungsi untuk menghitung matriks jarak DTW
def compute_dtw_distance_matrix(accumulated_cost_matrix):
    # Implementasi fungsi untuk menghitung matriks jarak DTW
    pass

# Halaman Pemetaan
def pemetaan():
    st.subheader("Pemetaan Clustering dengan DTW")

    # Unggah file data
    data_df = upload_csv_file(key="pemetaan_upload")

    if data_df is not None:
        # Proses data
        data_df['Tanggal'] = pd.to_datetime(data_df['Tanggal'], format='%d-%b-%y', errors='coerce')
        data_df.set_index('Tanggal', inplace=True)

        # Hitung rata-rata harian
        data_daily = data_df.resample('D').mean()

        # Standardisasi data harian
        scaler = StandardScaler()
        data_daily_standardized = pd.DataFrame(scaler.fit_transform(data_daily), index=data_daily.index, columns=data_daily.columns)

        # Hitung matriks jarak DTW
        local_cost_matrix_daily = compute_local_cost_matrix(data_daily_standardized)
        accumulated_cost_matrix_daily = compute_accumulated_cost_matrix(local_cost_matrix_daily)
        dtw_distance_matrix_daily = compute_dtw_distance_matrix(accumulated_cost_matrix_daily)

        # KMedoids Clustering
        max_n_clusters = 10
        silhouette_scores = {}
        for n_clusters in range(2, max_n_clusters + 1):
            kmedoids = KMedoids(n_clusters=n_clusters, metric='precomputed', method='pam', random_state=42)
            labels = kmedoids.fit_predict(dtw_distance_matrix_daily)
            silhouette_scores[n_clusters] = silhouette_score(dtw_distance_matrix_daily, labels, metric='precomputed')

        # Plot Silhouette Scores
        plt.figure(figsize=(10, 6))
        plt.plot(list(silhouette_scores.keys()), list(silhouette_scores.values()), marker='o', linestyle='-')
        plt.title('Silhouette Score vs. Number of Clusters (KMedoids)')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')
        plt.xticks(range(2, max_n_clusters + 1))
        plt.grid(True)
        st.pyplot(plt)

        # Menentukan jumlah kluster optimal
        optimal_n_clusters = max(silhouette_scores, key=silhouette_scores.get)
        st.write(f"Jumlah kluster optimal berdasarkan Silhouette Score (KMedoids) adalah: {optimal_n_clusters}")

        # Kluster labels untuk KMedoids
        cluster_labels = KMedoids(n_clusters=optimal_n_clusters, metric='precomputed', method='pam', random_state=42).fit_predict(dtw_distance_matrix_daily)

        # Memuat GeoJSON file dari GitHub
        gdf = upload_geojson_file()

        if gdf is not None:
            gdf = gdf.rename(columns={'Propinsi': 'Province'})  # Sesuaikan dengan nama kolom yang benar
            gdf['Province'] = gdf['Province'].str.upper().str.replace('.', '', regex=False).str.strip()

            # Mengganti nama provinsi yang tidak konsisten
            gdf['Province'] = gdf['Province'].replace({
                'DI ACEH': 'ACEH',
                'KEPULAUAN BANGKA BELITUNG': 'BANGKA BELITUNG',
                'NUSATENGGARA BARAT': 'NUSA TENGGARA BARAT',
                'D.I YOGYAKARTA': 'DI YOGYAKARTA',
                'DAERAH ISTIMEWA YOGYAKARTA': 'DI YOGYAKARTA',
            })

            # Menghapus provinsi yang None (i.e., GORONTALO)
            gdf = gdf[gdf['Province'].notna()]

            # Menampilkan daftar provinsi dalam GeoDataFrame untuk debugging
            st.write("Daftar Provinsi dalam GeoDataFrame:")
            st.write(gdf['Province'].tolist())

            # Menghitung data terklasifikasi
            clustered_data = pd.DataFrame({
                'Province': data_daily_standardized.columns,
                'Cluster': cluster_labels
            })

            clustered_data['Province'] = clustered_data['Province'].str.upper().str.replace('.', '', regex=False).str.strip()

            # Menggabungkan data terklasifikasi dengan GeoDataFrame
            gdf = gdf.merge(clustered_data, on='Province', how='left')

            # Mengatur warna untuk kluster
            gdf['color'] = gdf['Cluster'].map({
                0: 'red',
                1: 'yellow',
                2: 'green'
            })
            gdf['color'].fillna('grey', inplace=True)

            # Menampilkan provinsi yang berwarna grey
            grey_provinces = gdf[gdf['color'] == 'grey']['Province'].tolist()
            st.write("Provinsi yang tidak terklasifikasi:", grey_provinces)

            # Plot peta kluster menggunakan geopandas
            st.subheader("Peta Klustering Berdasarkan DTW")
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            gdf.boundary.plot(ax=ax, linewidth=1, color='black')
            gdf.plot(ax=ax, color=gdf['color'])
            plt.title("Peta Provinsi Berdasarkan Hasil Klustering DTW")
            plt.axis('off')
            st.pyplot(fig)

# Fungsi untuk melakukan clustering dengan complete linkage
def complete_linkage_clustering():
    st.subheader("Clustering dengan Complete Linkage")
    
    # Unggah file data
    data_df = upload_csv_file(key="clustering_upload")

    if data_df is not None:
        # Proses data sama seperti di atas
        data_df['Tanggal'] = pd.to_datetime(data_df['Tanggal'], format='%d-%b-%y', errors='coerce')
        data_df.set_index('Tanggal', inplace=True)
        
        # Hitung rata-rata harian
        data_daily = data_df.resample('D').mean()
        
        # Standardisasi data
        scaler = StandardScaler()
        data_daily_standardized = pd.DataFrame(scaler.fit_transform(data_daily), index=data_daily.index, columns=data_daily.columns)
        
        # Hitung matriks jarak DTW
        local_cost_matrix_daily = compute_local_cost_matrix(data_daily_standardized)
        accumulated_cost_matrix_daily = compute_accumulated_cost_matrix(local_cost_matrix_daily)
        dtw_distance_matrix_daily = compute_dtw_distance_matrix(accumulated_cost_matrix_daily)

        # Hierarchical Clustering dengan Complete Linkage
        from scipy.cluster.hierarchy import dendrogram, linkage
        
        Z = linkage(dtw_distance_matrix_daily, method='complete')
        plt.figure(figsize=(12, 8))
        dendrogram(Z, labels=data_daily.columns)
        plt.title("Dendrogram Hierarchical Clustering (Complete Linkage)")
        plt.xlabel("Provinces")
        plt.ylabel("Distance")
        st.pyplot(plt)

        # Menghitung kluster dari dendrogram (misalnya dengan potongan pada jarak tertentu)
        from scipy.cluster.hierarchy import fcluster
        max_d = 2  # Tentukan jarak maksimum untuk potong
        clusters = fcluster(Z, max_d, criterion='distance')

        # Memuat GeoJSON file dari GitHub
        gdf = upload_geojson_file()

        if gdf is not None:
            gdf = gdf.rename(columns={'Propinsi': 'Province'})
            gdf['Province'] = gdf['Province'].str.upper().str.replace('.', '', regex=False).str.strip()

            # Mengganti nama provinsi yang tidak konsisten
            gdf['Province'] = gdf['Province'].replace({
                'DI ACEH': 'ACEH',
                'KEPULAUAN BANGKA BELITUNG': 'BANGKA BELITUNG',
                'NUSATENGGARA BARAT': 'NUSA TENGGARA BARAT',
                'D.I YOGYAKARTA': 'DI YOGYAKARTA',
                'DAERAH ISTIMEWA YOGYAKARTA': 'DI YOGYAKARTA',
            })

            # Menghapus provinsi yang None (i.e., GORONTALO)
            gdf = gdf[gdf['Province'].notna()]

            # Menghitung data terklasifikasi
            clustered_data = pd.DataFrame({
                'Province': data_daily_standardized.columns,
                'Cluster': clusters
            })

            clustered_data['Province'] = clustered_data['Province'].str.upper().str.replace('.', '', regex=False).str.strip()

            # Menggabungkan data terklasifikasi dengan GeoDataFrame
            gdf = gdf.merge(clustered_data, on='Province', how='left')

            # Mengatur warna untuk kluster
            gdf['color'] = gdf['Cluster'].map({
                1: 'red',
                2: 'yellow',
                3: 'green'
            })
            gdf['color'].fillna('grey', inplace=True)

            # Menampilkan provinsi yang berwarna grey
            grey_provinces = gdf[gdf['color'] == 'grey']['Province'].tolist()
            st.write("Provinsi yang tidak terklasifikasi:", grey_provinces)

            # Plot peta kluster menggunakan geopandas
            st.subheader("Peta Klustering Berdasarkan Complete Linkage")
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            gdf.boundary.plot(ax=ax, linewidth=1, color='black')
            gdf.plot(ax=ax, color=gdf['color'])
            plt.title("Peta Provinsi Berdasarkan Hasil Klustering Complete Linkage")
            plt.axis('off')
            st.pyplot(fig)

# Fungsi utama untuk menjalankan aplikasi Streamlit
def main():
    st.title("Aplikasi Pemetaan Clustering")
    menu = ["Pemetaan KMedoids", "Clustering Complete Linkage"]
    choice = st.sidebar.selectbox("Pilih Menu", menu)

    if choice == "Pemetaan KMedoids":
        pemetaan()
    elif choice == "Clustering Complete Linkage":
        complete_linkage_clustering()

if __name__ == "__main__":
    main()
