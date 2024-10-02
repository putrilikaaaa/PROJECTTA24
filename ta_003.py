import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
import geopandas as gpd
from streamlit_option_menu import option_menu

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

# Standardization using the provided formulas
def standardize(data_df: pd.DataFrame) -> pd.DataFrame:
    n = len(data_df)
    Z_bar = data_df.mean()
    S_z = np.sqrt(((data_df - Z_bar) ** 2).mean())
    standardized_data = (data_df - Z_bar) / S_z
    return standardized_data

# Function to compute local cost matrix for DTW
def compute_local_cost_matrix(data_df: pd.DataFrame) -> np.array:
    num_time_points, num_provinces = data_df.shape
    local_cost_matrix = np.zeros((num_time_points, num_provinces, num_provinces))

    for i in range(num_provinces):
        for j in range(num_provinces):
            if i != j:
                for t in range(num_time_points):
                    local_cost_matrix[t, i, j] = np.abs(data_df.iloc[t, i] - data_df.iloc[t, j])

    return local_cost_matrix

# Function to compute accumulated cost matrix for DTW
def compute_accumulated_cost_matrix(local_cost_matrix: np.array) -> np.array:
    num_time_points, num_provinces, _ = local_cost_matrix.shape
    accumulated_cost_matrix = np.zeros((num_time_points, num_provinces, num_provinces))

    for i in range(num_provinces):
        for j in range(num_provinces):
            accumulated_cost_matrix[0, i, j] = local_cost_matrix[0, i, j]

    for t in range(1, num_time_points):
        for i in range(num_provinces):
            for j in range(num_provinces):
                if i == j:
                    accumulated_cost_matrix[t, i, j] = local_cost_matrix[t, i, j] + \
                        min(accumulated_cost_matrix[t-1, i, j], accumulated_cost_matrix[t-1, i-1, j], accumulated_cost_matrix[t-1, i+1, j])
                else:
                    accumulated_cost_matrix[t, i, j] = local_cost_matrix[t, i, j] + min(
                        accumulated_cost_matrix[t-1, i, j], accumulated_cost_matrix[t-1, i-1, j], accumulated_cost_matrix[t-1, i+1, j]
                    )

    return accumulated_cost_matrix

# Function to compute DTW distance matrix
def compute_dtw_distance_matrix(accumulated_cost_matrix: np.array) -> np.array:
    num_provinces = accumulated_cost_matrix.shape[1]
    dtw_distance_matrix = np.zeros((num_provinces, num_provinces))

    for i in range(num_provinces):
        for j in range(num_provinces):
            if i != j:
                dtw_distance_matrix[i, j] = accumulated_cost_matrix[-1, i, j]

    return dtw_distance_matrix

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
        province_options = [col for col in data_df.columns if col != 'Tanggal']  # Menghilangkan 'Tanggal' dari pilihan
        selected_province = st.selectbox("Pilih Provinsi untuk Visualisasi", province_options)

        if selected_province:
            # Visualisasi data untuk provinsi terpilih
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

        # Handle missing data by forward filling
        data_daily.fillna(method='ffill', inplace=True)

        # Standardize daily data
        data_daily_standardized = standardize(data_daily)

        # Dropdown for choosing linkage method
        linkage_method = st.selectbox("Pilih Metode Linkage", options=["complete", "single", "average"])

        # Compute local cost matrix and accumulated cost matrix
        local_cost_matrix_daily = compute_local_cost_matrix(data_daily_standardized)
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
            clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage=linkage_method)
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
        Z = linkage(condensed_dtw_distance_matrix, method=linkage_method)

        plt.figure(figsize=(16, 10))
        dendrogram(Z, labels=data_daily_standardized.columns, leaf_rotation=90)
        plt.title(f'Dendrogram Clustering dengan DTW (Data Harian) - Linkage: {linkage_method.capitalize()}')
        plt.xlabel('Provinsi')
        plt.ylabel('Jarak DTW')
        st.pyplot(plt)

        # Table of provinces per cluster
        cluster_labels = cluster_labels_dict[optimal_n_clusters]
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
            clustered_data['Province'] = clustered_data['Province'].str.upper()
            gdf = gdf.merge(clustered_data, on='Province', how='left')

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

# Sidebar Navigation
with st.sidebar:
    selected_page = option_menu("Menu", ["Statistika Deskriptif", "Pemetaan"], 
                                 icons=["info", "map"], 
                                 menu_icon="cast", 
                                 default_index=0)

# Main Application Logic
data_df = upload_csv_file()

if selected_page == "Statistika Deskriptif":
    statistika_deskriptif(data_df)
elif selected_page == "Pemetaan":
    pemetaan(data_df)
