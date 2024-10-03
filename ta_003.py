import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
import geopandas as gpd

# Fungsi untuk menghitung matriks biaya lokal
def compute_local_cost_matrix(data):
    return np.abs(data[:, None] - data[None, :])

# Fungsi untuk menghitung matriks biaya terakumulasi
def compute_accumulated_cost_matrix(local_cost_matrix):
    n = local_cost_matrix.shape[0]
    accumulated_cost = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == 0 and j == 0:
                accumulated_cost[i, j] = local_cost_matrix[i, j]
            elif i == 0:
                accumulated_cost[i, j] = accumulated_cost[i, j - 1] + local_cost_matrix[i, j]
            elif j == 0:
                accumulated_cost[i, j] = accumulated_cost[i - 1, j] + local_cost_matrix[i, j]
            else:
                accumulated_cost[i, j] = min(accumulated_cost[i - 1, j], accumulated_cost[i, j - 1]) + local_cost_matrix[i, j]
    return accumulated_cost

# Fungsi untuk menghitung matriks jarak DTW
def compute_dtw_distance_matrix(accumulated_cost_matrix):
    return accumulated_cost_matrix[-1, -1]

# Fungsi untuk menyimetrisasi matriks jarak
def symmetrize(matrix):
    if len(matrix.shape) == 1:
        matrix = np.expand_dims(matrix, axis=0)
    return (matrix + matrix.T) / 2

# Fungsi untuk menstandarisasi data
def standardize_data(data):
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)

# Fungsi untuk mengunggah file GeoJSON
def upload_geojson_file():
    geojson_file = st.file_uploader("Upload GeoJSON", type=["geojson"])
    if geojson_file is not None:
        return gpd.read_file(geojson_file)
    return None

# Fungsi Pemetaan
def pemetaan(data_df):
    st.subheader("Pemetaan Clustering dengan DTW")

    if data_df is not None:
        if 'Tanggal' in data_df.columns:
            data_df['Tanggal'] = pd.to_datetime(data_df['Tanggal'], format='%d-%b-%y', errors='coerce')
            data_df.set_index('Tanggal', inplace=True)

            # Menghitung rata-rata harian
            data_daily = data_df.resample('D').mean()
            data_daily.fillna(method='ffill', inplace=True)

            # Standardisasi data
            data_daily_values = standardize_data(data_daily.values)

            linkage_method = st.selectbox("Pilih Metode Linkage", options=["complete", "single", "average"])

            # Menghitung matriks biaya lokal
            local_cost_matrix_daily = compute_local_cost_matrix(data_daily_values)

            # Menghitung matriks biaya terakumulasi
            accumulated_cost_matrix_daily = compute_accumulated_cost_matrix(local_cost_matrix_daily)

            dtw_distance_matrix_daily = compute_dtw_distance_matrix(accumulated_cost_matrix_daily)

            # Memastikan dtw_distance_matrix_daily adalah array 2D sebelum disimetrisasi
            if dtw_distance_matrix_daily.ndim == 1:
                dtw_distance_matrix_daily = np.expand_dims(dtw_distance_matrix_daily, axis=0)

            dtw_distance_matrix_daily = symmetrize(dtw_distance_matrix_daily)

            num_samples = dtw_distance_matrix_daily.shape[0]

            max_n_clusters = 10
            silhouette_scores = {}
            cluster_labels_dict = {}

            for n_clusters in range(2, max_n_clusters + 1):
                clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage=linkage_method)
                labels = clustering.fit_predict(dtw_distance_matrix_daily)

                if len(labels) == num_samples:
                    try:
                        score = silhouette_score(dtw_distance_matrix_daily, labels, metric='precomputed')
                        silhouette_scores[n_clusters] = score
                        cluster_labels_dict[n_clusters] = labels
                    except ValueError as e:
                        st.error(f"Error calculating silhouette score: {e}")
                else:
                    st.error("Jumlah label tidak sesuai dengan ukuran matriks jarak.")

            if silhouette_scores:
                plt.figure(figsize=(10, 6))
                plt.plot(list(silhouette_scores.keys()), list(silhouette_scores.values()), marker='o', linestyle='-')

                for n_clusters, score in silhouette_scores.items():
                    plt.text(n_clusters, score, f"{score:.2f}", fontsize=9, ha='right')

                plt.title('Silhouette Score vs. Number of Clusters (Data Harian)')
                plt.xlabel('Number of Clusters')
                plt.ylabel('Silhouette Score')
                plt.xticks(range(2, max_n_clusters + 1))
                plt.grid(True)
                st.pyplot(plt)

            if silhouette_scores:
                optimal_n_clusters = max(silhouette_scores, key=silhouette_scores.get)
                st.write(f"Jumlah kluster optimal berdasarkan Silhouette Score adalah: {optimal_n_clusters}")

                condensed_dtw_distance_matrix = squareform(dtw_distance_matrix_daily)
                Z = linkage(condensed_dtw_distance_matrix, method=linkage_method)

                plt.figure(figsize=(16, 10))
                dendrogram(Z, labels=data_daily.columns, leaf_rotation=90)
                plt.title(f'Dendrogram Clustering dengan DTW (Data Harian) - Linkage: {linkage_method.capitalize()}')
                plt.xlabel('Provinsi')
                plt.ylabel('Jarak DTW')
                st.pyplot(plt)

                cluster_labels = cluster_labels_dict[optimal_n_clusters]
                clustered_data = pd.DataFrame({
                    'Province': data_daily.columns,
                    'Cluster': cluster_labels
                })

                st.subheader("Tabel Provinsi per Cluster")
                st.write(clustered_data)

                gdf = upload_geojson_file()

                if gdf is not None:
                    gdf = gdf.rename(columns={'Propinsi': 'Province'})
                    gdf['Province'] = gdf['Province'].str.upper().str.replace('.', '', regex=False).str.strip()

                    clustered_data['Province'] = clustered_data['Province'].str.upper().str.replace('.', '', regex=False).str.strip()

                    gdf['Province'] = gdf['Province'].replace({
                        'DI ACEH': 'ACEH',
                        'KEPULAUAN BANGKA BELITUNG': 'BANGKA BELITUNG',
                        'NUSATENGGARA BARAT': 'NUSA TENGGARA BARAT',
                        'D.I YOGYAKARTA': 'DI YOGYAKARTA',
                        'DAERAH ISTIMEWA YOGYAKARTA': 'DI YOGYAKARTA',
                    })

                    gdf = gdf[gdf['Province'].notna()]

                    gdf = gdf.merge(clustered_data, on='Province', how='left')

                    gdf['color'] = gdf['Cluster'].map({
                        0: 'red',
                        1: 'yellow',
                        2: 'green',
                        3: 'blue',
                        4: 'purple',
                        5: 'orange',
                        6: 'pink',
                        7: 'brown',
                        8: 'cyan',
                        9: 'magenta'
                    })
                    gdf['color'].fillna('grey', inplace=True)

                    grey_provinces = gdf[gdf['color'] == 'grey']['Province'].tolist()
                    if grey_provinces:
                        st.subheader("Provinsi yang Tidak Termasuk dalam Kluster:")
                        st.write(grey_provinces)
                    else:
                        st.write("Semua provinsi termasuk dalam kluster.")

                    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
                    gdf.boundary.plot(ax=ax, linewidth=1, color='black')
                    gdf.plot(ax=ax, color=gdf['color'], edgecolor='black', alpha=0.7)
                    plt.title("Pemetaan Provinsi Berdasarkan Kluster")
                    st.pyplot(fig)
        else:
            st.error("Kolom 'Tanggal' tidak ditemukan dalam data.")

# Fungsi utama
def main():
    st.title("Aplikasi Clustering Provinsi dengan DTW")

    # Upload file data
    uploaded_file = st.file_uploader("Upload Data", type=["csv"])
    if uploaded_file is not None:
        data_df = pd.read_csv(uploaded_file)

        # Pilih halaman
        page_selection = st.sidebar.radio("Pilih Halaman", ['Statistika Deskriptif', 'Pemetaan'])
        
        if page_selection == 'Statistika Deskriptif':
            # Tampilkan statistika deskriptif
            st.subheader("Statistika Deskriptif")
            st.write(data_df.describe())
        elif page_selection == 'Pemetaan':
            pemetaan(data_df)

if __name__ == "__main__":
    main()
