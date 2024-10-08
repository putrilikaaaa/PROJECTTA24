import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn_extra.cluster import KMedoids  # Importing KMedoids
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
import geopandas as gpd
from streamlit_option_menu import option_menu
from sklearn.preprocessing import MinMaxScaler  # Importing MinMaxScaler for normalization
from fastdtw import fastdtw  # Importing fastdtw for DTW computation

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

# Function to compute DTW distance matrix
def compute_dtw_distance_matrix(data):
    num_series = data.shape[1]
    dtw_distance_matrix = np.zeros((num_series, num_series))

    for i in range(num_series):
        for j in range(i, num_series):
            distance, _ = fastdtw(data[:, i], data[:, j])
            dtw_distance_matrix[i, j] = distance
            dtw_distance_matrix[j, i] = distance  # DTW distance is symmetric

    return dtw_distance_matrix

# Function to symmetrize a matrix (making it symmetric)
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
        province_options = [col for col in data_df.columns if col != 'Tanggal']  # Menghilangkan 'Tanggal' dari pilihan
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

# Pemetaan KMedoids Page
def pemetaan_kmedoids(data_df):
    st.subheader("Pemetaan KMedoids Clustering")

    if data_df is not None:
        data_df['Tanggal'] = pd.to_datetime(data_df['Tanggal'], format='%d-%b-%y', errors='coerce')
        data_df.set_index('Tanggal', inplace=True)

        # Calculate daily averages
        data_daily = data_df.resample('D').mean()

        # Handle missing data by forward filling
        data_daily.fillna(method='ffill', inplace=True)

        # Normalization of data (before computing DTW)
        scaler = MinMaxScaler()  # Using MinMaxScaler for normalization
        data_daily_values = scaler.fit_transform(data_daily)

        # Dropdown for choosing the number of clusters for KMedoids
        n_clusters = st.slider("Pilih Jumlah Kluster (KMedoids)", min_value=2, max_value=10, value=3)

        # Compute DTW distance matrix for normalized daily data using fastdtw
        dtw_distance_matrix_daily = compute_dtw_distance_matrix(data_daily_values)

        # Symmetrize the DTW distance matrix
        dtw_distance_matrix_daily = symmetrize(dtw_distance_matrix_daily)

        # KMedoids clustering
        kmedoids = KMedoids(n_clusters=n_clusters, metric="precomputed", random_state=42)
        cluster_labels = kmedoids.fit_predict(dtw_distance_matrix_daily)

        # Compute Silhouette Score
        silhouette_avg = silhouette_score(dtw_distance_matrix_daily, cluster_labels, metric='precomputed')
        st.write(f"Silhouette Score untuk {n_clusters} kluster: {silhouette_avg:.2f}")

        # Plot Silhouette Score
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(range(n_clusters), [silhouette_avg for _ in range(n_clusters)], align="center", alpha=0.7)
        ax.set_title("Silhouette Score per Cluster")
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Silhouette Score")
        st.pyplot(fig)

        # Load GeoJSON file from GitHub
        gdf = upload_geojson_file()

        if gdf is not None:
            gdf = gdf.rename(columns={'Propinsi': 'Province'})  # Change according to the correct column name
            gdf['Province'] = gdf['Province'].str.upper().str.replace('.', '', regex=False).str.strip()

            # Calculate cluster from clustering results
            cluster_data = pd.DataFrame({
                'Province': data_daily.columns,
                'Cluster': cluster_labels
            })

            # Merge with GeoDataFrame
            gdf = gdf.merge(cluster_data, on='Province', how='left')

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

            # Plot map
            fig, ax = plt.subplots(figsize=(10, 10))
            gdf.plot(ax=ax, color=gdf['color'])
            ax.set_title(f"Pemetaan Provinsi dengan KMedoids Clustering")
            st.pyplot(fig)

# Streamlit Sidebar for Navigation
def sidebar_navigation():
    with st.sidebar:
        selected = option_menu("Menu", ["Statistika Deskriptif", "Pemetaan KMedoids"],
                               icons=["bar-chart", "map"], menu_icon="cast", default_index=0)
    return selected

# Main function
def main():
    st.title("Streamlit Clustering Visualization")
    
    selected_page = sidebar_navigation()

    # Upload the data
    data = upload_csv_file()

    # Show the relevant page based on the selection
    if selected_page == "Statistika Deskriptif":
        statistika_deskriptif(data)
    elif selected_page == "Pemetaan KMedoids":
        pemetaan_kmedoids(data)

# Execute the application
if __name__ == "__main__":
    main()
