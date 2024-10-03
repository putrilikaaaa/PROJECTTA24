import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import squareform

# Function to compute local cost matrix
def compute_local_cost_matrix(data):
    # Your implementation of the local cost matrix calculation
    pass

# Function to compute accumulated cost matrix
def compute_accumulated_cost_matrix(local_cost_matrix):
    # Your implementation of the accumulated cost matrix calculation
    pass

# Function to compute DTW distance matrix
def compute_dtw_distance_matrix(accumulated_cost_matrix):
    # Your implementation of the DTW distance matrix calculation
    pass

# Function to symmetrize the distance matrix
def symmetrize(distance_matrix):
    return (distance_matrix + distance_matrix.T) / 2

# Function to upload GeoJSON file from GitHub
def upload_geojson_file():
    # Your implementation to load the GeoJSON file
    pass

# Pemetaan function
def pemetaan(data_df):
    st.subheader("Pemetaan Clustering dengan DTW")

    if data_df is not None:
        data_df['Tanggal'] = pd.to_datetime(data_df['Tanggal'], format='%d-%b-%y', errors='coerce')
        data_df.set_index('Tanggal', inplace=True)

        # Calculate daily averages
        data_daily = data_df.resample('D').mean()

        # Handle missing data by forward filling
        data_daily.fillna(method='ffill', inplace=True)

        # Standardization of data
        scaler = StandardScaler()
        data_daily_values = scaler.fit_transform(data_daily)

        # Compute local cost matrix
        local_cost_matrix_daily = compute_local_cost_matrix(pd.DataFrame(data_daily_values, columns=data_daily.columns))

        # Compute accumulated cost matrix
        accumulated_cost_matrix_daily = compute_accumulated_cost_matrix(local_cost_matrix_daily)

        # Compute DTW distance matrix for daily data
        dtw_distance_matrix_daily = compute_dtw_distance_matrix(accumulated_cost_matrix_daily)

        # Ensure DTW distance matrix is symmetric
        dtw_distance_matrix_daily = symmetrize(dtw_distance_matrix_daily)

        # Clustering and silhouette score calculation for daily data
        max_n_clusters = 10
        silhouette_scores = {}
        cluster_labels_dict = {}

        for n_clusters in range(2, max_n_clusters + 1):
            clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='average')
            labels = clustering.fit_predict(dtw_distance_matrix_daily)
            score = silhouette_score(dtw_distance_matrix_daily, labels, metric='precomputed')
            silhouette_scores[n_clusters] = score
            cluster_labels_dict[n_clusters] = labels

        # Plot Silhouette Scores
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
        optimal_n_clusters = max(silhouette_scores, key=silhouette_scores.get)
        st.write(f"Jumlah kluster optimal berdasarkan Silhouette Score adalah: {optimal_n_clusters}")

        # Clustering and dendrogram
        condensed_dtw_distance_matrix = squareform(dtw_distance_matrix_daily)
        Z = linkage(condensed_dtw_distance_matrix, method='average')

        # Ensure labels match the number of data points (provinces)
        labels = data_daily.columns.tolist()  # Get the names of the provinces

        plt.figure(figsize=(16, 10))
        dendrogram(Z, labels=labels, leaf_rotation=90)
        plt.title(f'Dendrogram Clustering dengan DTW (Data Harian) - Linkage: Average')
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
            st.write("Peta Clustering:")
            fig, ax = plt.subplots(1, 1, figsize=(12, 10))
            gdf['Cluster'] = cluster_labels  # Add cluster data to GeoDataFrame

            # Plotting the clusters on the map
            gdf.boundary.plot(ax=ax, linewidth=1, color='black')
            gdf.plot(column='Cluster', ax=ax, legend=True, cmap='RdYlGn')
            plt.title('Pemetaan Provinsi berdasarkan Kluster')
            plt.axis('off')
            st.pyplot(fig)
        else:
            st.error("GeoJSON file not found!")

# Main function to run the Streamlit application
def main():
    st.title("Aplikasi Pemetaan Clustering")
    
    # Upload data file
    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])
    
    if uploaded_file is not None:
        data_df = pd.read_csv(uploaded_file)
        pemetaan(data_df)

if __name__ == "__main__":
    main()
