import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import json

# Function to load GeoJSON from a URL
def load_geojson(url):
    return gpd.read_file(url)

# Function for data standardization
def standardize_data(data):
    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

# Function for clustering
def perform_clustering(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(data)
    return labels

# Main function for Streamlit app
def main():
    st.title("Aplikasi Pemetaan dan Analisis Data")

    # File upload section
    uploaded_file = st.file_uploader("Upload data CSV", type=["csv"])
    if uploaded_file is not None:
        data_df = pd.read_csv(uploaded_file)
        st.write(data_df.head())

        # Data preprocessing
        data_df['Tanggal'] = pd.to_datetime(data_df['Tanggal'], format='%d-%b-%y', errors='coerce')
        data_df.set_index('Tanggal', inplace=True)

        # Descriptive statistics
        st.subheader("Statistika Deskriptif")
        st.write(data_df.describe())

        # Select clustering parameters
        n_clusters = st.slider("Pilih jumlah kluster", min_value=2, max_value=10, value=3)

        # Standardize the data for clustering
        data_standardized = standardize_data(data_df)

        # Perform clustering
        cluster_labels = perform_clustering(data_standardized, n_clusters)

        # Prepare data for mapping
        cluster_labels_df = pd.DataFrame({
            'Province': data_standardized.columns,
            'Cluster': cluster_labels
        })

        # Load GeoJSON file from GitHub
        geojson_url = 'https://raw.githubusercontent.com/username/repo/branch/path/to/indonesia-prov.geojson'  # Replace with your actual URL
        gdf = load_geojson(geojson_url)

        # Check gdf structure
        st.write(gdf.columns)
        st.write(gdf.head())

        # Ensure proper naming in gdf and cluster_labels_df
        gdf['properties.name'] = gdf['properties.name'].str.upper().str.strip()
        cluster_labels_df['Province'] = cluster_labels_df['Province'].str.upper().str.strip()

        # Merge on the correct keys
        gdf = gdf.merge(cluster_labels_df, left_on='properties.name', right_on='Province', how='left')

        # Plotting clusters on the map
        st.subheader("Peta Provinsi dengan Cluster")
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        gdf.boundary.plot(ax=ax, linewidth=1, color="black")

        # Assign colors based on clusters
        color_map = {0: 'red', 1: 'yellow', 2: 'green'}
        gdf['Color'] = gdf['Cluster'].map(color_map)

        # Plot provinces with their clusters
        gdf.plot(column='Color', ax=ax, legend=True, missing_kwds={"color": "lightgrey"}, alpha=0.5)
        plt.title("Peta Kluster Provinsi")
        st.pyplot(fig)

# Run the app
if __name__ == "__main__":
    main()
