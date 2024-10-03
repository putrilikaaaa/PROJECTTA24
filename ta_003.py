import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import dendrogram, linkage
import geopandas as gpd

# Fungsi untuk standardisasi data
def standardize_data(data):
    return (data - data.mean()) / data.std()

# Fungsi untuk menghitung matriks jarak lokal (local cost matrix)
def compute_local_cost_matrix(data):
    # Contoh implementasi
    return np.abs(data[:, None] - data[None, :])

# Fungsi untuk menghitung matriks jarak terakumulasi (accumulated cost matrix)
def compute_accumulated_cost_matrix(local_cost_matrix):
    # Contoh implementasi
    return np.cumsum(local_cost_matrix, axis=1)

# Fungsi untuk menghitung matriks jarak DTW
def compute_dtw_distance_matrix(accumulated_cost_matrix):
    # Contoh implementasi
    return accumulated_cost_matrix[:, -1]

# Fungsi untuk memastikan matriks jarak simetris
def symmetrize(matrix):
    return (matrix + matrix.T) / 2

# Fungsi untuk mengunggah file GeoJSON
def upload_geojson_file():
    # Mengambil file GeoJSON dari GitHub (path lokal atau URL di sini)
    try:
        gdf = gpd.read_file("/path/to/indonesia-prov.geojson")
        return gdf
    except Exception as e:
        st.error(f"Error loading GeoJSON file: {e}")
        return None

# Pemetaan Page
def pemetaan(data_df):
    st.subheader("Pemetaan Clustering dengan DTW")

    if data_df is not None:
        data_df['Tanggal'] = pd.to_datetime(data_df['Tanggal'], format='%d-%b-%y', errors='coerce')
        data_df.set_index('Tanggal', inplace=True)

        # Calculate daily averages
        data_daily = data_df.resample('D').mean()

        # Handle missing data by forward filling
        data_daily.fillna(method='ffill', inplace=True)

        # **Standardization step**
        data_daily = standardize_data(data_daily)  # Apply standardization

        # No normalization, the data is already standardized
        data_daily_values = data_daily.values

        # Dropdown for choosing linkage method
        linkage_method = st.selectbox("Pilih Metode Linkage", options=["complete", "single", "average"])

        # Compute local cost matrix and accumulated cost matrix
        local_cost_matrix_daily = compute_local_cost_matrix(data_daily)
        accumulated_cost_matrix_daily = compute_accumulated_cost_matrix(local_cost_matrix_daily)

        # Compute DTW distance matrix for daily data
        dtw_distance_matrix_daily = compute_dtw_distance_matrix(accumulated_cost_matrix_daily)

        # Ensure DTW distance matrix is symmetric
        dtw_distance_matrix_daily = symmetrize(dtw_distance_matrix_daily)

        # **Check size of distance matrix and labels**
        num_samples = dtw_distance_matrix_daily.shape[0]
        
        # Clustering and silhouette score calculation for daily data
        max_n_clusters = 10
        silhouette_scores = {}
        cluster_labels_dict = {}

        for n_clusters in range(2, max_n_clusters + 1):
            clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage=linkage_method)
            labels = clustering.fit_predict(dtw_distance_matrix_daily)

            if len(labels) == num_samples:  # Ensure label size matches the distance matrix
                try:
                    score = silhouette_score(dtw_distance_matrix_daily, labels, metric='precomputed')
                    silhouette_scores[n_clusters] = score
                    cluster_labels_dict[n_clusters] = labels
                except ValueError as e:
                    st.error(f"Error calculating silhouette score: {e}")
            else:
                st.error("Jumlah label tidak sesuai dengan ukuran matriks jarak.")

        # Plot Silhouette Scores
        if silhouette_scores:
            plt.figure(figsize=(10, 6))
            plt.plot(list(silhouette_scores.keys()), list(silhouette_scores.values()), marker='o', linestyle='-')

            # Adding data labels to the silhouette score plot
            for n_clusters, score in silhouette_scores.items():
                plt.text(n_clusters, score, f"{score:.2f}", fontsize=9, ha='right')

            plt.title('Silhouette Score vs. Number of Clusters (Data Harian)')
            plt.xlabel('Number of Clusters')
            plt.ylabel('Silhouette Score')
            plt.xticks(range(2, max_n_clusters + 1))
            plt.grid(True)
            st.pyplot(plt)

        # Determine optimal number of clusters
        if silhouette_scores:
            optimal_n_clusters = max(silhouette_scores, key=silhouette_scores.get)
            st.write(f"Jumlah kluster optimal berdasarkan Silhouette Score adalah: {optimal_n_clusters}")
        
            # Clustering and dendrogram
            condensed_dtw_distance_matrix = squareform(dtw_distance_matrix_daily)
            Z = linkage(condensed_dtw_distance_matrix, method=linkage_method)

            plt.figure(figsize=(16, 10))
            dendrogram(Z, labels=data_daily.columns, leaf_rotation=90)
            plt.title(f'Dendrogram Clustering dengan DTW (Data Harian) - Linkage: {linkage_method.capitalize()}')
            plt.xlabel('Provinsi')
            plt.ylabel('Jarak DTW')
            st.pyplot(plt)

            # Table of provinces per cluster
            cluster_labels = cluster_labels_dict[optimal_n_clusters]
            clustered_data = pd.DataFrame({
                'Province': data_daily.columns,
                'Cluster': cluster_labels
            })

            # Display cluster table
            st.subheader("Tabel Provinsi per Cluster")
            st.write(clustered_data)

            # Load GeoJSON file from GitHub
            gdf = upload_geojson_file()

            if gdf is not None:
                gdf = gdf.rename(columns={'Propinsi': 'Province'})  # Change according to the correct column name
                gdf['Province'] = gdf['Province'].str.upper().str.replace('.', '', regex=False).str.strip()

                # Calculate cluster from clustering results
                clustered_data['Province'] = clustered_data['Province'].str.upper().str.replace('.', '', regex=False).str.strip()

                # Rename inconsistent provinces
                gdf['Province'] = gdf['Province'].replace({
                    'DI ACEH': 'ACEH',
                    'KEPULAUAN BANGKA BELITUNG': 'BANGKA BELITUNG',
                    'NUSATENGGARA BARAT': 'NUSA TENGGARA BARAT',
                    'D.I YOGYAKARTA': 'DI YOGYAKARTA',
                    'DAERAH ISTIMEWA YOGYAKARTA': 'DI YOGYAKARTA',
                })

                # Remove provinces that are None (i.e., GORONTALO)
                gdf = gdf[gdf['Province'].notna()]

                # Merge clustered data with GeoDataFrame
                gdf = gdf.merge(clustered_data, on='Province', how='left')

                # Set colors for clusters
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

                # Display provinces colored grey
                grey_provinces = gdf[gdf['color'] == 'grey']['Province'].tolist()
                if grey_provinces:
                    st.subheader("Provinsi yang Tidak Termasuk dalam Kluster:")
                    st.write(grey_provinces)
                else:
                    st.write("Semua provinsi termasuk dalam kluster.")

                # Plot map
                fig, ax = plt.subplots(1, 1, figsize=(12, 10))
                gdf.boundary.plot(ax=ax, linewidth=1, color='black')  # Plot boundaries
                gdf.plot(ax=ax, color=gdf['color'], edgecolor='black', alpha=0.7)  # Plot clusters
                plt.title("Pemetaan Provinsi Berdasarkan Kluster")
                st.pyplot(fig)

# Fungsi utama Streamlit
def main():
    st.title("Aplikasi Pemetaan Kluster dengan DTW")

    # Upload file data
    data_file = st.file_uploader("Unggah file CSV", type=["csv"])
    
    if data_file:
        data_df = pd.read_csv(data_file)
        pemetaan(data_df)

if __name__ == "__main__":
    main()
