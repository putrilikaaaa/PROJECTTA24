import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

# Function to compute local cost matrix
def compute_local_cost_matrix(data_df: pd.DataFrame) -> np.array:
    # Logic for computing the local cost matrix
    n = data_df.shape[0]
    cost_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            cost_matrix[i, j] = np.sum((data_df.iloc[i] - data_df.iloc[j]) ** 2)  # example: squared Euclidean distance
    return cost_matrix

def compute_accumulated_cost_matrix(local_cost_matrix: np.array) -> np.array:
    n = local_cost_matrix.shape[0]
    accumulated_cost_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if j == 0:
                accumulated_cost_matrix[i, j] = local_cost_matrix[i, j]
            else:
                accumulated_cost_matrix[i, j] = local_cost_matrix[i, j] + min(accumulated_cost_matrix[i, j - 1],
                                                                             accumulated_cost_matrix[i - 1, j])
    return accumulated_cost_matrix

def compute_dtw_distance_matrix(data_df: pd.DataFrame) -> np.array:
    local_cost_matrix = compute_local_cost_matrix(data_df)
    accumulated_cost_matrix = compute_accumulated_cost_matrix(local_cost_matrix)
    dtw_distance_matrix = accumulated_cost_matrix[-1, :, :]
    return dtw_distance_matrix

def symmetrize(matrix: np.array) -> np.array:
    # Ensure the distance matrix is symmetric
    return (matrix + matrix.T) / 2

# Pemetaan Page
def pemetaan(data_df):
    st.subheader("Pemetaan Clustering dengan DTW")

    if data_df is not None:
        data_df['Tanggal'] = pd.to_datetime(data_df['Tanggal'], format='%d-%b-%y', errors='coerce')
        data_df.set_index('Tanggal', inplace=True)

        # Calculate daily averages
        data_daily = data_df.resample('D').mean()

        # Standardize daily data
        scaler = StandardScaler()
        data_daily_standardized = pd.DataFrame(scaler.fit_transform(data_daily), index=data_daily.index, columns=data_daily.columns)

        # Compute DTW distance matrix for daily data
        dtw_distance_matrix_daily = compute_dtw_distance_matrix(data_daily_standardized)

        # Ensure DTW distance matrix is symmetric
        dtw_distance_matrix_daily = symmetrize(dtw_distance_matrix_daily)

        # Verify the shape of the distance matrix
        st.write("Shape of DTW Distance Matrix:", dtw_distance_matrix_daily.shape)

        # Clustering and silhouette score calculation for daily data
        max_n_clusters = 10
        silhouette_scores = {}

        for n_clusters in range(2, max_n_clusters + 1):
            clustering = AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed', linkage='average')
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
        Z = linkage(condensed_dtw_distance_matrix, method='average')

        plt.figure(figsize=(16, 10))
        dendrogram(Z, labels=data_daily_standardized.columns, leaf_rotation=90)
        plt.title(f'Dendrogram Clustering dengan DTW (Data Harian)')
        plt.xlabel('Provinsi')
        plt.ylabel('Jarak DTW')
        st.pyplot(plt)

        # Table of provinces per cluster
        cluster_labels = AgglomerativeClustering(n_clusters=optimal_n_clusters, affinity='precomputed', linkage='average').fit_predict(dtw_distance_matrix_daily)
        clustered_data = pd.DataFrame({
            'Province': data_daily_standardized.columns,
            'Cluster': cluster_labels
        })

        # Display cluster table
        st.subheader("Tabel Provinsi per Cluster")
        st.write(clustered_data)

# Main function
def main():
    st.title("Aplikasi Clustering dengan DTW")
    
    # Upload data file
    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])
    if uploaded_file is not None:
        data_df = pd.read_csv(uploaded_file)
        pemetaan(data_df)

if __name__ == "__main__":
    main()
