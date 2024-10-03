import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from sklearn_extra.cluster import KMedoids
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

        # Standardization of data
        scaler = StandardScaler()
        data_daily_values = scaler.fit_transform(data_daily)

        # Dropdown for choosing the number of clusters
        n_clusters = st.slider("Pilih Jumlah Kluster", min_value=2, max_value=10, value=3)

        # Compute local cost matrix and accumulated cost matrix
        local_cost_matrix_daily = compute_local_cost_matrix(pd.DataFrame(data_daily_values, columns=data_daily.columns))
        accumulated_cost_matrix_daily = compute_accumulated_cost_matrix(local_cost_matrix_daily)

        # Compute DTW distance matrix for daily data
        dtw_distance_matrix_daily = compute_dtw_distance_matrix(accumulated_cost_matrix_daily)

        # Ensure DTW distance matrix is symmetric
        dtw_distance_matrix_daily = symmetrize(dtw_distance_matrix_daily)

        # Clustering with K-Medoids
        kmedoids = KMedoids(n_clusters=n_clusters, metric='precomputed', random_state=42)
        labels = kmedoids.fit_predict(dtw_distance_matrix_daily)

        # Calculate silhouette score
        score = silhouette_score(dtw_distance_matrix_daily, labels, metric='precomputed')
        st.write(f"Silhouette Score untuk K-Medoids dengan {n_clusters} kluster: {score:.2f}")

        # Dendrogram
        condensed_dtw_distance_matrix = squareform(dtw_distance_matrix_daily)
        Z = linkage(condensed_dtw_distance_matrix, method='complete')

        plt.figure(figsize=(16, 10))
        dendrogram(Z, labels=data_daily.columns, leaf_rotation=90)
        plt.title('Dendrogram Clustering K-Medoids')
        plt.xlabel('Provinsi')
        plt.ylabel('Jarak DTW')
        st.pyplot(plt)

        # Table of provinces per cluster
        clustered_data = pd.DataFrame({
            'Province': data_daily.columns,
            'Cluster': labels
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
            plt.title("Pemetaan Provinsi Berdasarkan Kluster K-Medoids")
            st.pyplot(fig)

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
    num_time_points, num_provinces = local_cost_matrix.shape[0], local_cost_matrix.shape[1]
    accumulated_cost_matrix = np.zeros((num_time_points, num_provinces, num_provinces))

    for t in range(1, num_time_points):
        for i in range(num_provinces):
            for j in range(num_provinces):
                accumulated_cost_matrix[t, i, j] = local_cost_matrix[t, i, j] + min(
                    accumulated_cost_matrix[t - 1, i, j],  # from the same province
                    accumulated_cost_matrix[t - 1, j, i],  # from the other province
                    accumulated_cost_matrix[t - 1, i, i]   # from the diagonal
                )

    return accumulated_cost_matrix

# Function to compute DTW distance matrix
def compute_dtw_distance_matrix(accumulated_cost_matrix: np.array) -> np.array:
    num_provinces = accumulated_cost_matrix.shape[1]
    dtw_distance_matrix = np.zeros((num_provinces, num_provinces))

    for i in range(num_provinces):
        for j in range(num_provinces):
            dtw_distance_matrix[i, j] = accumulated_cost_matrix[-1, i, j]

    return dtw_distance_matrix

# Main application
def main():
    st.title("Aplikasi Pemetaan Data")
    
    # Sidebar menu
    with st.sidebar:
        selected = option_menu("Menu", ["Statistika Deskriptif", "Pemetaan KMedoids"],
                               icons=["file-earmark-text", "geo-alt"], 
                               menu_icon="cast", default_index=0)

    # Upload data
    data_df = upload_csv_file()

    # Render the selected page
    if selected == "Statistika Deskriptif":
        statistika_deskriptif(data_df)
    elif selected == "Pemetaan KMedoids":
        pemetaan_kmedoids(data_df)

if __name__ == "__main__":
    main()
