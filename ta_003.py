import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from sklearn_extra.cluster import KMedoids
from scipy.cluster.hierarchy import dendrogram, linkage
from dtaidistance import dtw
import geopandas as gpd

# Function definitions for DTW calculations
def compute_local_cost_matrix(data):
    num_series = data.shape[1]
    local_cost_matrix = np.zeros((num_series, num_series))

    for i in range(num_series):
        for j in range(num_series):
            local_cost_matrix[i, j] = dtw.distance(data.iloc[:, i], data.iloc[:, j])

    return local_cost_matrix

def compute_accumulated_cost_matrix(local_cost_matrix):
    num_series = local_cost_matrix.shape[0]
    accumulated_cost_matrix = np.zeros_like(local_cost_matrix)

    for i in range(num_series):
        for j in range(num_series):
            if i == 0 and j == 0:
                accumulated_cost_matrix[i, j] = local_cost_matrix[i, j]
            elif i == 0:
                accumulated_cost_matrix[i, j] = accumulated_cost_matrix[i, j - 1] + local_cost_matrix[i, j]
            elif j == 0:
                accumulated_cost_matrix[i, j] = accumulated_cost_matrix[i - 1, j] + local_cost_matrix[i, j]
            else:
                accumulated_cost_matrix[i, j] = min(
                    accumulated_cost_matrix[i - 1, j], 
                    accumulated_cost_matrix[i, j - 1], 
                    accumulated_cost_matrix[i - 1, j - 1]
                ) + local_cost_matrix[i, j]

    return accumulated_cost_matrix

def compute_dtw_distance_matrix(accumulated_cost_matrix):
    num_series = accumulated_cost_matrix.shape[0]
    dtw_distance_matrix = np.zeros((num_series, num_series))

    for i in range(num_series):
        for j in range(num_series):
            dtw_distance_matrix[i, j] = accumulated_cost_matrix[i, j] / (i + j + 1)

    return dtw_distance_matrix

# Streamlit application
def main():
    st.title("Analisis DTW dan Clustering")

    # File upload
    uploaded_file = st.file_uploader("Upload data CSV", type=["csv"])
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        
        # Preprocessing: Standardize the data
        scaler = StandardScaler()
        data_standardized = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

        # Compute local cost matrix
        local_cost_matrix = compute_local_cost_matrix(data_standardized)

        # Compute accumulated cost matrix
        accumulated_cost_matrix = compute_accumulated_cost_matrix(local_cost_matrix)

        # Compute DTW distance matrix
        dtw_distance_matrix = compute_dtw_distance_matrix(accumulated_cost_matrix)

        # Display the DTW distance matrix
        st.write("Matriks Jarak DTW:")
        st.dataframe(dtw_distance_matrix)

        # Clustering
        n_clusters = st.slider("Pilih jumlah cluster", min_value=2, max_value=10, value=3)
        clustering_method = st.selectbox("Pilih metode clustering", ["Agglomerative Clustering", "K-Medoids"])

        if clustering_method == "Agglomerative Clustering":
            clustering_model = AgglomerativeClustering(n_clusters=n_clusters)
        else:
            clustering_model = KMedoids(n_clusters=n_clusters)

        clusters = clustering_model.fit_predict(dtw_distance_matrix)
        st.write("Hasil Clustering:")
        st.dataframe(clusters)

        # Silhouette Score
        silhouette_avg = silhouette_score(dtw_distance_matrix, clusters)
        st.write(f"Silhouette Score: {silhouette_avg:.2f}")

        # Plot dendrogram
        linked = linkage(dtw_distance_matrix, 'ward')
        plt.figure(figsize=(10, 7))
        dendrogram(linked, orientation='top', labels=data.columns, distance_sort='descending', show_leaf_counts=True)
        plt.title('Dendrogram')
        st.pyplot(plt)

if __name__ == "__main__":
    main()
