import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn_extra.cluster import KMedoids
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
import geopandas as gpd

# Dummy implementation of DTW local cost matrix
def compute_local_cost_matrix(data):
    # Here, you should implement the actual DTW cost calculation
    return np.abs(data.values[:, None] - data.values[None, :])  # Simple example for illustration

# Dummy implementation of DTW accumulated cost matrix
def compute_accumulated_cost_matrix(local_cost_matrix):
    # Here, implement the logic to calculate the accumulated cost
    n = local_cost_matrix.shape[0]
    accumulated_cost = np.zeros_like(local_cost_matrix)
    accumulated_cost[0, :] = np.cumsum(local_cost_matrix[0, :])
    for i in range(1, n):
        for j in range(n):
            accumulated_cost[i, j] = local_cost_matrix[i, j] + min(accumulated_cost[i-1, j], accumulated_cost[i, j-1])
    return accumulated_cost

# Dummy implementation of DTW distance matrix
def compute_dtw_distance_matrix(accumulated_cost_matrix):
    # Implement the logic to extract DTW distances
    return accumulated_cost_matrix[-1, -1]  # Example for illustration

# Function to upload CSV files
def upload_csv_file(key=None):
    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"], key=key)
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            return df
        except Exception as e:
            st.error(f"Error: {e}")
    return None

# Function to upload GeoJSON files
def upload_geojson_file():
    gdf = gpd.read_file('https://raw.githubusercontent.com/putrilikaaaa/PROJECTTA24/main/indonesia-prov.geojson')
    return gdf

# Descriptive Statistics Page
def statistik_deskriptif():
    st.subheader("Statistika Deskriptif")
    data_df = upload_csv_file()

    if data_df is not None:
        st.write("Dataframe:")
        st.write(data_df)

        # Convert 'Tanggal' column to datetime and set as index
        data_df['Tanggal'] = pd.to_datetime(data_df['Tanggal'], format='%d-%b-%y', errors='coerce')
        data_df.set_index('Tanggal', inplace=True)

        if data_df.isnull().any().any():
            st.warning("Terdapat tanggal yang tidak valid, silakan periksa data Anda.")

        # Dropdown for selecting province
        selected_province = st.selectbox("Pilih Provinsi", options=data_df.columns.tolist())

        if selected_province:
            # Display descriptive statistics
            st.subheader(f"Statistika Deskriptif untuk {selected_province}")
            st.write(data_df[selected_province].describe())

            # Display line chart
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

# Mapping Page
def pemetaan():
    st.subheader("Pemetaan Clustering dengan DTW")
    data_df = upload_csv_file(key="pemetaan_upload")

    if data_df is not None:
        data_df['Tanggal'] = pd.to_datetime(data_df['Tanggal'], format='%d-%b-%y', errors='coerce')
        data_df.set_index('Tanggal', inplace=True)

        # Calculate daily averages
        data_daily = data_df.resample('D').mean()

        # Standardize daily data
        scaler = StandardScaler()
        data_daily_standardized = pd.DataFrame(scaler.fit_transform(data_daily), index=data_daily.index, columns=data_daily.columns)

        # Compute local cost matrix and accumulated cost matrix
        local_cost_matrix_daily = compute_local_cost_matrix(data_daily_standardized)
        accumulated_cost_matrix_daily = compute_accumulated_cost_matrix(local_cost_matrix_daily)

        # Compute DTW distance matrix for daily data
        dtw_distance_matrix_daily = compute_dtw_distance_matrix(accumulated_cost_matrix_daily)

        # Dropdown for selecting clustering method
        clustering_method = st.selectbox("Pilih Metode Clustering", ["Complete Linkage", "K-Medoids"])

        if clustering_method == "Complete Linkage":
            # Clustering and silhouette score calculation for daily data
            max_n_clusters = 10
            silhouette_scores = {}

            for n_clusters in range(2, max_n_clusters + 1):
                clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='complete')
                labels = clustering.fit_predict(dtw_distance_matrix_daily)
                score = silhouette_score(dtw_distance_matrix_daily, labels, metric='precomputed')
                silhouette_scores[n_clusters] = score

            # Plot Silhouette Scores
            plt.figure(figsize=(10, 6))
            plt.plot(list(silhouette_scores.keys()), list(silhouette_scores.values()), marker='o', linestyle='-')
            plt.title('Silhouette Score vs. Number of Clusters (Data Harian)')
            plt.xlabel('Number of Clusters')
            plt.ylabel('Silhouette Score')
            plt.xticks(range(2, max_n_clusters + 1))
            plt.grid(True)
            st.pyplot(plt)

            # Determine optimal number of clusters
            optimal_n_clusters = max(silhouette_scores, key=silhouette_scores.get)
            st.write(f"Jumlah kluster optimal berdasarkan Silhouette Score adalah: {optimal_n_clusters}")

            # Clustering and dendrogram
            condensed_dtw_distance_matrix = squareform(dtw_distance_matrix_daily)
            Z = linkage(condensed_dtw_distance_matrix, method='complete')

            plt.figure(figsize=(16, 10))
            dendrogram(Z, labels=data_daily_standardized.columns, leaf_rotation=90)
            plt.title('Dendrogram Clustering dengan DTW (Data Harian)')
            plt.xlabel('Provinsi')
            plt.ylabel('Jarak DTW')
            st.pyplot(plt)

            # Table of provinces per cluster
            cluster_labels = AgglomerativeClustering(n_clusters=optimal_n_clusters, metric='precomputed', linkage='complete').fit_predict(dtw_distance_matrix_daily)
            clustered_data = pd.DataFrame({
                'Province': data_daily_standardized.columns,
                'Cluster': cluster_labels
            })

        elif clustering_method == "K-Medoids":
            # K-Medoids clustering and silhouette score calculation
            max_n_clusters = 10
            silhouette_scores_kmedoids = {}

            for n_clusters in range(2, max_n_clusters + 1):
                clustering = KMedoids(n_clusters=n_clusters, metric='precomputed')
                labels = clustering.fit_predict(dtw_distance_matrix_daily)
                score = silhouette_score(dtw_distance_matrix_daily, labels, metric='precomputed')
                silhouette_scores_kmedoids[n_clusters] = score

            # Plot Silhouette Scores for K-Medoids
            plt.figure(figsize=(10, 6))
            plt.plot(list(silhouette_scores_kmedoids.keys()), list(silhouette_scores_kmedoids.values()), marker='o', linestyle='-')
            plt.title('Silhouette Score vs. Number of Clusters (K-Medoids)')
            plt.xlabel('Number of Clusters')
            plt.ylabel('Silhouette Score')
            plt.xticks(range(2, max_n_clusters + 1))
            plt.grid(True)
            st.pyplot(plt)

            # Determine optimal number of clusters for K-Medoids
            optimal_n_clusters_kmedoids = max(silhouette_scores_kmedoids, key=silhouette_scores_kmedoids.get)
            st.write(f"Jumlah kluster optimal berdasarkan Silhouette Score (K-Medoids) adalah: {optimal_n_clusters_kmedoids}")

            # Clustering result with K-Medoids
            cluster_labels_kmedoids = KMedoids(n_clusters=optimal_n_clusters_kmedoids, metric='precomputed').fit_predict(dtw_distance_matrix_daily)
            clustered_data = pd.DataFrame({
                'Province': data_daily_standardized.columns,
                'Cluster': cluster_labels_kmedoids
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

            # Rename inconsistent provinces in gdf
            replacements = {
                'DI ACEH': 'ACEH',
                'GORONTALO': None,  # This will drop GORONTALO
                'KEPULAUAN BANGKA BELITUNG': 'BANGKA BELITUNG',
                'NUSATENGGARA BARAT': 'NUSA TENGGARA BARAT',
                'D.I YOGYAKARTA': 'DI YOGYAKARTA',
                'DAERAH ISTIMEWA YOGYAKARTA': 'D.I YOGYAKARTA'
            }

            gdf.replace({'Province': replacements}, inplace=True)

            # Merge GeoDataFrame with clustered data
            gdf = gdf.merge(clustered_data, on='Province', how='left')

            # Assign colors to clusters
            gdf['color'] = gdf['Cluster'].map({0: 'red', 1: 'yellow', 2: 'green'})
            grey_provinces = gdf[gdf['color'].isna()]['Province'].unique()
            st.write("Provinsi tanpa Cluster:", grey_provinces)

            # Plot clustering map
            st.subheader("Peta Clustering")
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            gdf.boundary.plot(ax=ax, linewidth=1)
            gdf.plot(column='color', ax=ax, legend=True, alpha=0.5, missing_kwds={'color': 'lightgrey'})
            plt.title('Peta Clustering berdasarkan DTW')
            plt.axis('off')
            st.pyplot(fig)

# Main Application
def main():
    st.title("Aplikasi Clustering Provinsi")
    page = st.sidebar.selectbox("Pilih Halaman", ["Statistika Deskriptif", "Pemetaan"])

    if page == "Statistika Deskriptif":
        statistik_deskriptif()
    elif page == "Pemetaan":
        pemetaan()

if __name__ == "__main__":
    main()
