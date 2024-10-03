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
from sklearn.preprocessing import StandardScaler

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

# Function to compute local cost matrix for DTW
def compute_local_cost_matrix(data):
    n = data.shape[0]
    local_cost_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            # Assuming we compute the distance as the absolute difference for simplicity
            local_cost_matrix[i, j] = np.sum(np.abs(data.iloc[i] - data.iloc[j]))
    
    return local_cost_matrix

# Ensure DTW distance matrix is symmetric
def symmetrize(matrix):
    return (matrix + matrix.T) / 2

# Function to compute accumulated cost matrix
def compute_accumulated_cost_matrix(local_cost_matrix):
    n = local_cost_matrix.shape[0]
    accumulated_cost_matrix = np.zeros((n, n))
    accumulated_cost_matrix[0, 0] = local_cost_matrix[0, 0]
    
    for i in range(1, n):
        accumulated_cost_matrix[i, 0] = accumulated_cost_matrix[i-1, 0] + local_cost_matrix[i, 0]
        accumulated_cost_matrix[0, i] = accumulated_cost_matrix[0, i-1] + local_cost_matrix[0, i]
    
    for i in range(1, n):
        for j in range(1, n):
            accumulated_cost_matrix[i, j] = min(
                accumulated_cost_matrix[i-1, j],
                accumulated_cost_matrix[i, j-1],
                accumulated_cost_matrix[i-1, j-1]
            ) + local_cost_matrix[i, j]
    
    return accumulated_cost_matrix

# Function to compute DTW distance matrix
def compute_dtw_distance_matrix(accumulated_cost_matrix):
    n = accumulated_cost_matrix.shape[0]
    dtw_distance_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            dtw_distance_matrix[i, j] = accumulated_cost_matrix[i, j]
    
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

        # Convert 'Tanggal' to datetime format
        data_df['Tanggal'] = pd.to_datetime(data_df['Tanggal'], format='%d-%b-%y', errors='coerce')
        
        # Dropdown untuk memilih provinsi, kecuali kolom 'Tanggal'
        province_options = [col for col in data_df.columns if col != 'Tanggal']  # Menghilangkan 'Tanggal' dari pilihan
        selected_province = st.selectbox("Pilih Provinsi untuk Visualisasi", province_options)

        # Calendar date picker for selecting a date
        selected_date = st.date_input("Pilih Tanggal:", value=data_df['Tanggal'].min(), min_value=data_df['Tanggal'].min(), max_value=data_df['Tanggal'].max())

        if selected_date:
            # Filter data for the selected date
            daily_data = data_df[data_df['Tanggal'] == pd.to_datetime(selected_date)]

            if not daily_data.empty:
                # Plot average prices for the selected date for all provinces
                fig, ax = plt.subplots(figsize=(10, 5))
                daily_values = daily_data.iloc[0, 1:]  # Get the first row's values except 'Tanggal'
                ax.bar(daily_values.index, daily_values.values, color='skyblue')
                ax.set_title(f"Nilai pada Tanggal {selected_date}")
                ax.set_xlabel("Provinsi")
                ax.set_ylabel("Nilai")
                ax.set_xticklabels(daily_values.index, rotation=45)
                st.pyplot(fig)
            else:
                st.write(f"Tidak ada data untuk tanggal {selected_date}.")

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

        # Make sure to use the correct number of labels (number of original provinces)
        labels = data_daily.columns

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
            gdf.plot(column='Cluster', ax=ax, legend=True, cmap='RdYlGn', edgecolor='black')
            ax.set_title("Peta Clustering Berdasarkan DTW")
            plt.axis('off')
            st.pyplot(fig)

# Sidebar Menu
with st.sidebar:
    selected_option = option_menu("Menu", ["Statistika Deskriptif", "Pemetaan"], 
                                   icons=["bar-chart", "map"], menu_icon="cast", default_index=0)

# Main function to control navigation between pages
def main():
    data_df = upload_csv_file()
    if data_df is not None:
        if selected_option == "Statistika Deskriptif":
            statistika_deskriptif(data_df)
        elif selected_option == "Pemetaan":
            pemetaan(data_df)
    else:
        st.warning("Silakan unggah file CSV untuk melanjutkan.")

# Run the app
if __name__ == "__main__":
    main()
