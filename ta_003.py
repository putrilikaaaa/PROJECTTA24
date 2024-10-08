import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn_extra.cluster import KMedoids
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import folium
from streamlit_folium import st_folium

# Function to compute DTW distance matrix
def compute_dtw_distance_matrix(data):
    n = data.shape[1]  # Number of time series (columns)
    distance_matrix = np.zeros((n, n))  # Initialize a square matrix

    for i in range(n):
        for j in range(i + 1, n):
            # Ensure that each time series is a 1D array
            series_i = data[:, i].ravel()  # Convert to 1D array
            series_j = data[:, j].ravel()  # Convert to 1D array
            distance, _ = fastdtw(series_i, series_j, dist=euclidean)  # Compute DTW distance
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance  # Symmetric matrix

    return distance_matrix

# Function to perform clustering using KMedoids
def perform_clustering(distance_matrix, n_clusters):
    model = KMedoids(n_clusters=n_clusters, metric='precomputed', random_state=0)
    model.fit(distance_matrix)
    return model.labels_

# Standardize data function
def standardize_data(data):
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(data)
    return standardized_data

# Function to visualize clustering on map
def visualize_clustering(geojson, labels, provinces):
    # Create a map centered around Indonesia
    m = folium.Map(location=[-2.5, 118.0], zoom_start=5)

    # Color settings for clusters
    cluster_colors = ['red', 'yellow', 'green']
    
    for idx, province in enumerate(provinces):
        province_geo = geojson[geojson['Propinsi'] == province]
        if not province_geo.empty:
            # Assign color based on the cluster
            folium.GeoJson(
                province_geo,
                style_function=lambda x, color=cluster_colors[labels[idx]]: {
                    'fillColor': color,
                    'color': 'black',
                    'weight': 1,
                    'fillOpacity': 0.7
                }
            ).add_to(m)
    
    return m

# Main function
def main():
    st.title("Clustering Visualizer with DTW Distance and KMedoids")

    # File upload
    uploaded_file = st.file_uploader("Upload your data", type=["csv"])
    if uploaded_file is not None:
        # Load data
        data_df = pd.read_csv(uploaded_file)

        # Process data for DTW and clustering
        pemetaan(data_df)

# Pemetaan function
def pemetaan(data_df):
    # Select relevant columns for clustering (skip the 'Tanggal' column)
    data_daily_values = data_df.iloc[:, 1:]

    # Step 1: Standardize the data
    standardized_data = standardize_data(data_daily_values)

    # Step 2: Compute DTW distance matrix
    dtw_distance_matrix_daily = compute_dtw_distance_matrix(standardized_data.T)

    # Step 3: Perform clustering (Example: 3 clusters)
    labels = perform_clustering(dtw_distance_matrix_daily, n_clusters=3)

    # Load GeoJSON file from GitHub or local storage
    geojson_path = '/content/indonesia-prov.geojson'
    geojson_data = gpd.read_file(geojson_path)

    # Ensure province names are consistent
    province_names = data_df.columns[1:]  # Extract province names from columns, excluding 'Tanggal'

    # Step 4: Visualize the clustering results on a map
    clustering_map = visualize_clustering(geojson_data, labels, province_names)

    # Display the map in Streamlit
    st_folium(clustering_map, width=700)

if __name__ == "__main__":
    main()
