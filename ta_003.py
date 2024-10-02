import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
import geopandas as gpd
from streamlit_option_menu import option_menu
from fastdtw import fastdtw  # Import fastdtw untuk perhitungan DTW

# Function to upload CSV files
def upload_csv_file():
    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])
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

# Ensure DTW distance matrix is symmetric
def symmetrize(matrix):
    return (matrix + matrix.T) / 2

# Statistika Deskriptif Page
def statistika_deskriptif(data_df):
    st.subheader("Statistika Deskriptif")

    if data_df is not None:
        st.write("Data yang diunggah:")
        st.write(data_df)

        # Statistika deskriptif
        st.write("Statistika deskriptif data:")
        st.write(data_df.describe())

        # Dropdown untuk memilih provinsi, kecuali kolom 'Tanggal'
        province_options = [col for col in data_df.columns if col != 'Tanggal']
        selected_province = st.selectbox("Pilih Provinsi untuk Visualisasi", province_options)

        if selected_province:
            st.write(f"Rata-rata harga untuk provinsi: {selected_province}")
            data_df['Tanggal'] = pd.to_datetime(data_df['Tanggal'], format='%d-%b-%y', errors='coerce')
            data_df.set_index('Tanggal', inplace=True)

            # Plot average prices for the selected province
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(data_df.index, data_df[selected_province], label=selected_province)
            ax.set_title(f"Rata-rata Harga Harian - Provinsi {selected_province}")
            ax.set_xlabel("Tanggal")
            ax.set_ylabel("Harga")
            ax.legend()

            st.pyplot(fig)

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

        # Dropdown for choosing linkage method
        linkage_method = st.selectbox("Pilih Metode Linkage", options=["complete", "single", "average"])

        # Compute local cost matrix and accumulated cost matrix
        local_cost_matrix_daily = compute_local_cost_matrix(data_daily_standardized)

        # Compute DTW distance matrix for daily data
        dtw_distance_matrix_daily = squareform(local_cost_matrix_daily)

        # Ensure DTW distance matrix is symmetric
        dtw_distance_matrix_daily = symmetrize(dtw_distance_matrix_daily)

        # Clustering and silhouette score calculation for daily data
        max_n_clusters = 10
        silhouette_scores = {}

        for n_clusters in range(2, max_n_clusters + 1):
            clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage=linkage_method)
            labels = clustering.fit_predict(dtw_distance_matrix_daily)

            # Calculate silhouette score
            if len(set(labels)) > 1:  # Ensure there is more than one cluster
                score = silhouette_score(dtw_distance_matrix_daily, labels, metric='precomputed')
                silhouette_scores[n_clusters] = score

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
        if silhouette_scores:
            optimal_n_clusters = max(silhouette_scores, key=silhouette_scores.get)
            st.write(f"Jumlah kluster optimal berdasarkan Silhouette Score adalah: {optimal_n_clusters}")

            # Clustering and dendrogram
            condensed_dtw_distance_matrix = squareform(dtw_distance_matrix_daily)
            Z = linkage(condensed_dtw_distance_matrix, method=linkage_method)

            plt.figure(figsize=(16, 10))
            dendrogram(Z, labels=data_daily_standardized.columns, leaf_rotation=90)
            plt.title(f'Dendrogram Clustering dengan DTW (Data Harian) - Linkage: {linkage_method.capitalize()}')
            plt.xlabel('Provinsi')
            plt.ylabel('Jarak DTW')
            st.pyplot(plt)

            # Table of provinces per cluster
            cluster_labels = AgglomerativeClustering(n_clusters=optimal_n_clusters, metric='precomputed', linkage=linkage_method).fit_predict(dtw_distance_matrix_daily)
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
                st.pyplot(plt)
            else:
                st.warning("Silakan upload file GeoJSON dengan benar.")

# Function to compute local cost matrix for DTW
def compute_local_cost_matrix(data):
    n_series = data.shape[1]
    local_cost_matrix = np.zeros((n_series, n_series))

    for i in range(n_series):
        for j in range(n_series):
            # Calculate DTW distance for each pair of series
            if i != j:
                local_cost_matrix[i, j] = fastdtw(data.iloc[:, i].values, data.iloc[:, j].values)[0]

    return local_cost_matrix

# Function to compute accumulated cost matrix for DTW
def compute_accumulated_cost_matrix(local_cost_matrix):
    n = local_cost_matrix.shape[0]
    accumulated_cost_matrix = np.zeros((n, n))
    accumulated_cost_matrix[0, 0] = local_cost_matrix[0, 0]

    for i in range(1, n):
        accumulated_cost_matrix[i, 0] = accumulated_cost_matrix[i - 1, 0] + local_cost_matrix[i, 0]
        accumulated_cost_matrix[0, i] = accumulated_cost_matrix[0, i - 1] + local_cost_matrix[0, i]

    for i in range(1, n):
        for j in range(1, n):
            accumulated_cost_matrix[i, j] = min(
                accumulated_cost_matrix[i - 1, j] + local_cost_matrix[i, j],
                accumulated_cost_matrix[i, j - 1] + local_cost_matrix[i, j],
                accumulated_cost_matrix[i - 1, j - 1] + 2 * local_cost_matrix[i, j]
            )

    return accumulated_cost_matrix

# Streamlit Layout
def main():
    st.title("Aplikasi Clustering dengan DTW")

    selected = option_menu(
        menu_title="Menu",
        options=["Statistika Deskriptif", "Pemetaan"],
        icons=["bar-chart", "geo-alt"],
        default_index=0
    )

    # Upload CSV data
    data_df = upload_csv_file()

    if selected == "Statistika Deskriptif":
        statistika_deskriptif(data_df)
    elif selected == "Pemetaan":
        pemetaan(data_df)

if __name__ == "__main__":
    main()
