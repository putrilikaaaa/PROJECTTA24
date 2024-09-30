import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn_extra.cluster import KMedoids
import geopandas as gpd

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

# Mapping Page with KMedoids
def pemetaan_kmedoids():
    st.subheader("Pemetaan Clustering dengan KMedoids")
    data_df = upload_csv_file(key="pemetaan_kmedoids_upload")

    if data_df is not None:
        data_df['Tanggal'] = pd.to_datetime(data_df['Tanggal'], format='%d-%b-%y', errors='coerce')
        data_df.set_index('Tanggal', inplace=True)

        # Calculate daily averages
        data_daily = data_df.resample('D').mean()

        # Standardize daily data
        scaler = StandardScaler()
        data_daily_standardized = pd.DataFrame(scaler.fit_transform(data_daily), index=data_daily.index, columns=data_daily.columns)

        # Apply KMedoids clustering
        max_n_clusters = 10
        silhouette_scores = {}

        for n_clusters in range(2, max_n_clusters + 1):
            kmedoids = KMedoids(n_clusters=n_clusters, metric='euclidean', method='pam', random_state=0)
            labels = kmedoids.fit_predict(data_daily_standardized.T)  # Transpose for province-wise clustering
            score = silhouette_score(data_daily_standardized.T, labels)
            silhouette_scores[n_clusters] = score

        # Plot Silhouette Scores
        plt.figure(figsize=(10, 6))
        plt.plot(list(silhouette_scores.keys()), list(silhouette_scores.values()), marker='o', linestyle='-')
        plt.title('Silhouette Score vs. Number of Clusters (KMedoids)')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')
        plt.xticks(range(2, max_n_clusters + 1))
        plt.grid(True)
        st.pyplot(plt)

        # Determine optimal number of clusters
        optimal_n_clusters = max(silhouette_scores, key=silhouette_scores.get)
        st.write(f"Jumlah kluster optimal berdasarkan Silhouette Score adalah: {optimal_n_clusters}")

        # Apply KMedoids with the optimal number of clusters
        kmedoids = KMedoids(n_clusters=optimal_n_clusters, metric='euclidean', method='pam', random_state=0)
        cluster_labels = kmedoids.fit_predict(data_daily_standardized.T)

        # Table of provinces per cluster
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
            gdf = gdf.rename(columns={'Propinsi': 'Province'})  # Adjust to the correct column name
            gdf['Province'] = gdf['Province'].str.upper().str.replace('.', '', regex=False).str.strip()

            # Rename inconsistent provinces
            clustered_data['Province'] = clustered_data['Province'].str.upper().str.replace('.', '', regex=False).str.strip()

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
            plt.title('Peta Kluster Provinsi di Indonesia (KMedoids)', fontsize=15)
            plt.xlabel('Longitude', fontsize=12)
            plt.ylabel('Latitude', fontsize=12)
            st.pyplot(plt)
        else:
            st.warning("Silakan upload file GeoJSON.")

# Main Function for Streamlit App
def main():
    st.title("Aplikasi Clustering dengan DTW, KMedoids, dan Visualisasi Pemetaan")
    menu = ["Statistika Deskriptif", "Pemetaan", "Pemetaan dengan KMedoids"]
    choice = st.sidebar.selectbox("Pilih Menu", menu)

    if choice == "Statistika Deskriptif":
        statistik_deskriptif()
    elif choice == "Pemetaan":
        pemetaan()
    elif choice == "Pemetaan dengan KMedoids":
        pemetaan_kmedoids()

if __name__ == "__main__":
    main()
