import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn_extra.cluster import KMedoids  # Import K-Medoids
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

# Fungsi untuk menghitung matriks biaya lokal
def compute_local_cost_matrix(data):
    # Implementasi fungsi ini sesuai kebutuhan
    pass

# Fungsi untuk menghitung matriks biaya terakumulasi
def compute_accumulated_cost_matrix(local_cost_matrix):
    # Implementasi fungsi ini sesuai kebutuhan
    pass

# Fungsi untuk menghitung matriks jarak DTW
def compute_dtw_distance_matrix(accumulated_cost_matrix):
    # Implementasi fungsi ini sesuai kebutuhan
    pass

# Fungsi untuk mengupload file CSV
def upload_csv_file(key):
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key=key)
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return None

# Fungsi untuk mengupload file GeoJSON
def upload_geojson_file():
    # Implementasi untuk mengupload GeoJSON dari repositori GitHub
    pass

# Mapping Page
def pemetaan():
    st.subheader("Pemetaan Clustering dengan DTW")
    data_df = upload_csv_file(key="pemetaan_upload")

    if data_df is not None:
        data_df['Tanggal'] = pd.to_datetime(data_df['Tanggal'], format='%d-%b-%y', errors='coerce')
        data_df.set_index('Tanggal', inplace=True)

        # Hitung rata-rata harian
        data_daily = data_df.resample('D').mean()

        # Standarisasi data harian
        scaler = StandardScaler()
        data_daily_standardized = pd.DataFrame(scaler.fit_transform(data_daily), index=data_daily.index, columns=data_daily.columns)

        # Hitung matriks biaya lokal dan matriks biaya terakumulasi
        local_cost_matrix_daily = compute_local_cost_matrix(data_daily_standardized)
        accumulated_cost_matrix_daily = compute_accumulated_cost_matrix(local_cost_matrix_daily)

        # Hitung matriks jarak DTW untuk data harian
        dtw_distance_matrix_daily = compute_dtw_distance_matrix(accumulated_cost_matrix_daily)

        # Klustering dan perhitungan skor siluet untuk data harian
        max_n_clusters = 10
        silhouette_scores_agglomerative = {}
        silhouette_scores_kmedoids = {}

        # Agglomerative Clustering
        for n_clusters in range(2, max_n_clusters + 1):
            clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='complete')
            labels = clustering.fit_predict(dtw_distance_matrix_daily)
            score = silhouette_score(dtw_distance_matrix_daily, labels, metric='precomputed')
            silhouette_scores_agglomerative[n_clusters] = score

        # K-Medoids Clustering
        for n_clusters in range(2, max_n_clusters + 1):
            kmedoids = KMedoids(n_clusters=n_clusters, metric='precomputed')
            labels = kmedoids.fit_predict(dtw_distance_matrix_daily)
            score = silhouette_score(dtw_distance_matrix_daily, labels, metric='precomputed')
            silhouette_scores_kmedoids[n_clusters] = score

        # Plot Silhouette Scores untuk Agglomerative dan K-Medoids
        plt.figure(figsize=(10, 6))
        plt.plot(list(silhouette_scores_agglomerative.keys()), list(silhouette_scores_agglomerative.values()), marker='o', linestyle='-', label='Agglomerative')
        plt.plot(list(silhouette_scores_kmedoids.keys()), list(silhouette_scores_kmedoids.values()), marker='s', linestyle='--', label='K-Medoids')
        plt.title('Silhouette Score vs. Number of Clusters (Data Harian)')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')
        plt.xticks(range(2, max_n_clusters + 1))
        plt.grid(True)
        plt.legend()
        st.pyplot(plt)

        # Tentukan jumlah kluster optimal untuk Agglomerative
        optimal_n_clusters_agglomerative = max(silhouette_scores_agglomerative, key=silhouette_scores_agglomerative.get)
        st.write(f"Jumlah kluster optimal untuk Agglomerative berdasarkan Silhouette Score adalah: {optimal_n_clusters_agglomerative}")

        # Tentukan jumlah kluster optimal untuk K-Medoids
        optimal_n_clusters_kmedoids = max(silhouette_scores_kmedoids, key=silhouette_scores_kmedoids.get)
        st.write(f"Jumlah kluster optimal untuk K-Medoids berdasarkan Silhouette Score adalah: {optimal_n_clusters_kmedoids}")

        # Klustering dan dendrogram untuk Agglomerative
        condensed_dtw_distance_matrix = squareform(dtw_distance_matrix_daily)
        Z = linkage(condensed_dtw_distance_matrix, method='complete')

        plt.figure(figsize=(16, 10))
        dendrogram(Z, labels=data_daily_standardized.columns, leaf_rotation=90)
        plt.title('Dendrogram Clustering dengan DTW (Data Harian)')
        plt.xlabel('Provinsi')
        plt.ylabel('Jarak DTW')
        st.pyplot(plt)

        # Tabel provinsi per kluster untuk Agglomerative
        cluster_labels_agglomerative = AgglomerativeClustering(n_clusters=optimal_n_clusters_agglomerative, metric='precomputed', linkage='complete').fit_predict(dtw_distance_matrix_daily)
        clustered_data_agglomerative = pd.DataFrame({
            'Province': data_daily_standardized.columns,
            'Cluster': cluster_labels_agglomerative
        })

        # Tabel provinsi per kluster untuk K-Medoids
        cluster_labels_kmedoids = KMedoids(n_clusters=optimal_n_clusters_kmedoids, metric='precomputed').fit_predict(dtw_distance_matrix_daily)
        clustered_data_kmedoids = pd.DataFrame({
            'Province': data_daily_standardized.columns,
            'Cluster': cluster_labels_kmedoids
        })

        # Tampilkan tabel kluster untuk Agglomerative
        st.subheader("Tabel Provinsi per Cluster (Agglomerative)")
        st.write(clustered_data_agglomerative)

        # Tampilkan tabel kluster untuk K-Medoids
        st.subheader("Tabel Provinsi per Cluster (K-Medoids)")
        st.write(clustered_data_kmedoids)

        # Load GeoJSON file dari GitHub
        gdf = upload_geojson_file()

        if gdf is not None:
            gdf = gdf.rename(columns={'Propinsi': 'Province'})  # Ubah sesuai dengan nama kolom yang benar
            gdf['Province'] = gdf['Province'].str.upper().str.replace('.', '', regex=False).str.strip()

            # Hitung cluster dari hasil klustering
            clustered_data_agglomerative['Province'] = clustered_data_agglomerative['Province'].str.upper().str.replace('.', '', regex=False).str.strip()
            clustered_data_kmedoids['Province'] = clustered_data_kmedoids['Province'].str.upper().str.replace('.', '', regex=False).str.strip()

            # Ganti nama provinsi yang tidak konsisten
            gdf['Province'] = gdf['Province'].replace({
                'DI ACEH': 'ACEH',
                'KEPULAUAN BANGKA BELITUNG': 'BANGKA BELITUNG',
                'NUSATENGGARA BARAT': 'NUSA TENGGARA BARAT',
                'D.I YOGYAKARTA': 'DI YOGYAKARTA',
                'DAERAH ISTIMEWA YOGYAKARTA': 'DI YOGYAKARTA',
            })

            # Hapus provinsi yang None (yaitu, GORONTALO)
            gdf = gdf[gdf['Province'].notna()]

            # Gabungkan data kluster dengan GeoDataFrame untuk Agglomerative
            gdf_agglomerative = gdf.merge(clustered_data_agglomerative, on='Province', how='left')
            gdf_kmedoids = gdf.merge(clustered_data_kmedoids, on='Province', how='left')

            # Set warna untuk cluster
            gdf_agglomerative['color'] = gdf_agglomerative['Cluster'].map({
                0: 'red',
                1: 'yellow',
                2: 'green'
            })
            gdf_agglomerative['color'].fillna('grey', inplace=True)

            gdf_kmedoids['color'] = gdf_kmedoids['Cluster'].map({
                0: 'red',
                1: 'yellow',
                2: 'green'
            })
            gdf_kmedoids['color'].fillna('grey', inplace=True)

            # Tampilkan provinsi yang berwarna abu-abu untuk Agglomerative
            grey_provinces_agglomerative = gdf_agglomerative[gdf_agglomerative['color'] == 'grey']['Province'].tolist()
            if grey_provinces_agglomerative:
                st.subheader("Provinsi yang Tidak Termasuk dalam Kluster (Agglomerative):")
                st.write(grey_provinces_agglomerative)
            else:
                st.write("Semua provinsi termasuk dalam kluster (Agglomerative).")

            # Tampilkan provinsi yang berwarna abu-abu untuk K-Medoids
            grey_provinces_kmedoids = gdf_kmedoids[gdf_kmedoids['color'] == 'grey']['Province'].tolist()
            if grey_provinces_kmedoids:
                st.subheader("Provinsi yang Tidak Termasuk dalam Kluster (K-Medoids):")
                st.write(grey_provinces_kmedoids)
            else:
                st.write("Semua provinsi termasuk dalam kluster (K-Medoids).")

            # Plot peta untuk Agglomerative
            fig_agglomerative, ax_agglomerative = plt.subplots(1, 1, figsize=(12, 10))
            gdf_agglomerative.boundary.plot(ax=ax_agglomerative, linewidth=1, color='black')  # Plot batas
            gdf_agglomerative.plot(ax=ax_agglomerative, color=gdf_agglomerative['color'], edgecolor='black', alpha=0.6)  # Plot provinsi dengan warna
            plt.title('Peta Kluster Provinsi di Indonesia (Agglomerative)', fontsize=15)
            st.pyplot(fig_agglomerative)

            # Plot peta untuk K-Medoids
            fig_kmedoids, ax_kmedoids = plt.subplots(1, 1, figsize=(12, 10))
            gdf_kmedoids.boundary.plot(ax=ax_kmedoids, linewidth=1, color='black')  # Plot batas
            gdf_kmedoids.plot(ax=ax_kmedoids, color=gdf_kmedoids['color'], edgecolor='black', alpha=0.6)  # Plot provinsi dengan warna
            plt.title('Peta Kluster Provinsi di Indonesia (K-Medoids)', fontsize=15)
            st.pyplot(fig_kmedoids)

# Menjalankan aplikasi Streamlit
if __name__ == "__main__":
    st.title("Aplikasi Pemetaan Clustering")
    pemetaan()
