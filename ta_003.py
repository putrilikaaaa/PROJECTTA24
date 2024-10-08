import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn_extra.cluster import KMedoids
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform, euclidean
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

# Compute Local Cost Matrix (Euclidean distance)
def compute_local_cost_matrix(data):
    n = data.shape[1]  # Number of provinces (columns)
    cost_matrix = np.zeros((n, n))
    
    # Calculate pairwise Euclidean distance between columns
    for i in range(n):
        for j in range(i, n):
            cost_matrix[i, j] = cost_matrix[j, i] = euclidean(data.iloc[:, i], data.iloc[:, j])
    return cost_matrix

# Compute Accumulated Cost Matrix
def compute_accumulated_cost_matrix(cost_matrix):
    n = cost_matrix.shape[0]
    accumulated_cost_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i, n):
            if i == 0 and j == 0:
                accumulated_cost_matrix[i, j] = cost_matrix[i, j]
            elif i == 0:
                accumulated_cost_matrix[i, j] = accumulated_cost_matrix[i, j-1] + cost_matrix[i, j]
            elif j == 0:
                accumulated_cost_matrix[i, j] = accumulated_cost_matrix[i-1, j] + cost_matrix[i, j]
            else:
                accumulated_cost_matrix[i, j] = min(
                    accumulated_cost_matrix[i-1, j], 
                    accumulated_cost_matrix[i, j-1], 
                    accumulated_cost_matrix[i-1, j-1]
                ) + cost_matrix[i, j]
    return accumulated_cost_matrix

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

        # Standardization of data
        scaler = StandardScaler()
        data_daily_values = scaler.fit_transform(data_daily)

        # Dropdown for choosing linkage method or KMedoids
        clustering_method = st.selectbox("Pilih Metode Clustering", options=["complete", "single", "average", "KMedoids"])

        # Compute local cost matrix and accumulated cost matrix
        local_cost_matrix_daily = compute_local_cost_matrix(pd.DataFrame(data_daily_values, columns=data_daily.columns))
        accumulated_cost_matrix_daily = compute_accumulated_cost_matrix(local_cost_matrix_daily)

        # Compute DTW distance matrix for daily data
        dtw_distance_matrix_daily = compute_dtw_distance_matrix(accumulated_cost_matrix_daily)

        # Ensure DTW distance matrix is symmetric
        dtw_distance_matrix_daily = symmetrize(dtw_distance_matrix_daily)

        # Clustering and silhouette score calculation for daily data
        max_n_clusters = 10
        silhouette_scores = {}
        cluster_labels_dict = {}

        if clustering_method in ["complete", "single", "average"]:
            # Agglomerative Clustering
            for n_clusters in range(2, max_n_clusters + 1):
                clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage=clustering_method)
                labels = clustering.fit_predict(dtw_distance_matrix_daily)
                score = silhouette_score(dtw_distance_matrix_daily, labels, metric='precomputed')
                silhouette_scores[n_clusters] = score
                cluster_labels_dict[n_clusters] = labels
        elif clustering_method == "KMedoids":
            # KMedoids Clustering
            for n_clusters in range(2, max_n_clusters + 1):
                clustering = KMedoids(n_clusters=n_clusters, metric='precomputed', init='k-medoids++')
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
        Z = linkage(condensed_dtw_distance_matrix, method=clustering_method)

        plt.figure(figsize=(16, 10))
        dendrogram(Z, labels=data_daily.columns, leaf_rotation=90)
        plt.title(f'Dendrogram Clustering dengan DTW (Data Harian) - Linkage: {clustering_method.capitalize()}')
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
                'DAERAH ISTIMEWA YOGYAKARTA': 'D.I YOGYAKARTA'
            })

            # Merge GeoDataFrame with cluster labels
            gdf = gdf.merge(clustered_data, on='Province')

            # Create map for visualization
            fig, ax = plt.subplots(1, 1, figsize=(15, 15))
            gdf.plot(column='Cluster', cmap='RdYlGn', edgecolor='black', linewidth=0.7, ax=ax)
            ax.set_title("Pemetaan Kluster Provinsi Indonesia")
            st.pyplot(fig)

# Main Streamlit App Layout
def main():
    st.title("Aplikasi Clustering DTW - Pemetaan Provinsi Indonesia")

    st.sidebar.title("Menu")
    selected_page = option_menu(
        menu_title=None,
        options=["Statistika Deskriptif", "Pemetaan"],
        icons=["graph-up", "map"],
        default_index=0,
        orientation="horizontal",
    )

    data_df = upload_csv_file()

    if selected_page == "Statistika Deskriptif":
        statistika_deskriptif(data_df)
    elif selected_page == "Pemetaan":
        pemetaan(data_df)

if __name__ == "__main__":
    main()
