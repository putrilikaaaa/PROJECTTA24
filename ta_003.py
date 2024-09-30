import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from kmedoids import KMedoids  # Ensure this import works
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
import geopandas as gpd
import lzma
import pickle

# Function to compute local cost matrix
def compute_local_cost_matrix(data):
    # Your implementation here...
    pass

# Function to compute accumulated cost matrix
def compute_accumulated_cost_matrix(local_cost_matrix):
    # Your implementation here...
    pass

# Function to compute DTW distance matrix
def compute_dtw_distance_matrix(accumulated_cost_matrix):
    # Your implementation here...
    pass

# Function to upload CSV file
def upload_csv_file(key):
    uploaded_file = st.file_uploader("Upload File CSV", type=["csv"], key=key)
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        return data
    return None

# Function to upload GeoJSON file
def upload_geojson_file():
    uploaded_geojson = st.file_uploader("Upload File GeoJSON", type=["geojson"])
    if uploaded_geojson is not None:
        gdf = gpd.read_file(uploaded_geojson)
        return gdf
    return None

# Main function
def main():
    st.title("Aplikasi Pemetaan dan Analisis Klustering")
    menu = ["Pemetaan", "Analisis"]
    choice = st.sidebar.selectbox("Pilih Halaman", menu)

    if choice == "Pemetaan":
        pemetaan()
    elif choice == "Analisis":
        st.write("Fitur analisis akan ditambahkan.")

# Function for mapping
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
        if local_cost_matrix_daily is not None:
            accumulated_cost_matrix_daily = compute_accumulated_cost_matrix(local_cost_matrix_daily)

            # Compute DTW distance matrix for daily data
            dtw_distance_matrix_daily = compute_dtw_distance_matrix(accumulated_cost_matrix_daily)

            # Dropdown for selecting clustering method
            clustering_method = st.selectbox("Pilih Metode Clustering", ["Complete Linkage", "K-Medoids"])

            # Clustering and silhouette score calculation for daily data
            max_n_clusters = 10
            silhouette_scores = {}

            for n_clusters in range(2, max_n_clusters + 1):
                if clustering_method == "Complete Linkage":
                    clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='complete')
                    labels = clustering.fit_predict(dtw_distance_matrix_daily)
                elif clustering_method == "K-Medoids":
                    clustering = KMedoids(n_clusters=n_clusters, metric='precomputed', random_state=42)
                    labels = clustering.fit_predict(dtw_distance_matrix_daily)

                # Check if the labels are valid
                if len(set(labels)) > 1:  # Ensure there is more than one cluster
                    try:
                        score = silhouette_score(dtw_distance_matrix_daily, labels, metric='precomputed')
                        silhouette_scores[n_clusters] = score
                    except ValueError as ve:
                        st.warning(f"ValueError for {n_clusters} clusters: {ve}")
                else:
                    st.warning(f"Cannot compute silhouette score for {n_clusters} clusters, only one cluster detected.")

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
            cluster_labels = (AgglomerativeClustering(n_clusters=optimal_n_clusters, metric='precomputed', linkage='complete' if clustering_method == "Complete Linkage" else 'average')
                              .fit_predict(dtw_distance_matrix_daily))
            clustered_data = pd.DataFrame({
                'Province': data_daily_standardized.columns,
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
                    2: 'green'
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
                gdf.plot(ax=ax, color=gdf['color'], edgecolor='black', alpha=0.6)  # Plot provinces with colors

                # Add title and labels
                plt.title('Peta Kluster Provinsi di Indonesia', fontsize=15)
                plt.xlabel('Longitude', fontsize=12)
                plt.ylabel('Latitude', fontsize=12)

                # Show plot in Streamlit
                st.pyplot(fig)

# Run the app
if __name__ == '__main__':
    main()
