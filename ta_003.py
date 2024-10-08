import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn_extra.cluster import KMedoids
import geopandas as gpd
import json
import requests

# Function to load GeoJSON from GitHub
def upload_geojson_file():
    geojson_url = 'https://raw.githubusercontent.com/your-repo/indonesia-prov.geojson'  # Replace with actual GitHub URL
    try:
        response = requests.get(geojson_url)
        if response.status_code == 200:
            geojson_data = response.json()
            gdf = gpd.read_file(json.dumps(geojson_data))
            return gdf
        else:
            st.error("Failed to load GeoJSON data from GitHub.")
            return None
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

# Data Upload page
def upload_data():
    st.subheader("Upload Data")
    uploaded_file = st.file_uploader("Pilih file CSV untuk diupload", type="csv")

    if uploaded_file is not None:
        data_df = pd.read_csv(uploaded_file)
        st.write("Data yang berhasil diupload:", data_df.head())
        return data_df
    return None

# Pemetaan Page with clustering method selection and map visualization
def pemetaan(data_df):
    st.subheader("Pemetaan")

    if data_df is not None:
        # Ensure 'Tanggal' column is in datetime format
        data_df['Tanggal'] = pd.to_datetime(data_df['Tanggal'], format='%d-%b-%y', errors='coerce')
        data_df.set_index('Tanggal', inplace=True)

        # Dropdown to select a province, excluding 'Tanggal'
        province_options = [col for col in data_df.columns if col != 'Tanggal']
        selected_province = st.selectbox("Pilih Provinsi untuk Visualisasi", province_options)

        if selected_province:
            st.write(f"Rata-rata harga harian untuk provinsi: {selected_province}")

            # Extract the selected province data
            selected_data = data_df[[selected_province]].dropna()  # Drop NaN values

            if selected_data.empty:
                st.warning(f"Tidak ada data untuk provinsi {selected_province}")
            else:
                st.write(selected_data)

                # Plot the data
                st.line_chart(selected_data)

        # Clustering method selection
        st.sidebar.subheader("Pilih Metode Klastering")
        clustering_method = st.sidebar.radio("Metode Klastering", ["KMeans", "DBSCAN", "KMedoids"])

        # Standardization of data for clustering
        data_df_std = data_df.apply(pd.to_numeric, errors='coerce').fillna(0)
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_df_std)

        # KMeans clustering
        if clustering_method == "KMeans":
            n_clusters = st.sidebar.slider("Jumlah Klaster", min_value=2, max_value=10, value=3)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(data_scaled.T)  # Transpose so each column (province) is a sample
            data_df["Cluster"] = clusters

        # DBSCAN clustering
        elif clustering_method == "DBSCAN":
            epsilon = st.sidebar.slider("Epsilon", 0.01, 0.1, 0.05)
            min_samples = st.sidebar.slider("Min Samples", 2, 10, 5)
            dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
            clusters = dbscan.fit_predict(data_scaled.T)
            data_df["Cluster"] = clusters

        # KMedoids clustering
        elif clustering_method == "KMedoids":
            n_clusters = st.sidebar.slider("Jumlah Klaster", min_value=2, max_value=10, value=3)
            kmedoids = KMedoids(n_clusters=n_clusters, random_state=42, metric='euclidean')
            clusters = kmedoids.fit_predict(data_scaled.T)
            data_df["Cluster"] = clusters

        # Display the cluster results
        st.write(f"Hasil Klastering dengan metode {clustering_method}")
        st.write(data_df)

        # Load GeoJSON file from GitHub
        gdf = upload_geojson_file()

        if gdf is not None:
            gdf = gdf.rename(columns={'Propinsi': 'Province'})  # Change according to the correct column name
            gdf['Province'] = gdf['Province'].str.upper().str.replace('.', '', regex=False).str.strip()

            # Calculate cluster from clustering results
            data_df['Province'] = data_df.index.str.upper().str.replace('.', '', regex=False).str.strip()

            # Merge the clustered data with GeoDataFrame
            gdf = gdf.merge(data_df[['Province', 'Cluster']], on='Province', how='left')

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
            plt.title(f"Pemetaan Provinsi Berdasarkan Kluster ({clustering_method})")
            st.pyplot(fig)

# Statistika Deskriptif page
def statistika_deskriptif(data_df):
    st.subheader("Statistika Deskriptif")
    st.write("Data Deskriptif untuk Provinsi")

    if data_df is not None:
        st.write(data_df.describe())

# Main function to render the app
def main():
    st.title("Aplikasi Analisis Klastering Provinsi")

    # Upload Data
    data_df = upload_data()

    # Select Page
    st.sidebar.title("Navigasi")
    page = st.sidebar.selectbox("Pilih Halaman", ["Statistika Deskriptif", "Pemetaan"])

    if page == "Statistika Deskriptif":
        statistika_deskriptif(data_df)
    elif page == "Pemetaan":
        pemetaan(data_df)

if __name__ == "__main__":
    main()
