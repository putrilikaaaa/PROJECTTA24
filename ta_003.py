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
from sklearn.preprocessing import MinMaxScaler  # Importing MinMaxScaler for normalization
from fastdtw import fastdtw  # Importing fastdtw for DTW computation
from sklearn_extra.cluster import KMedoids  # Importing KMedoids for clustering

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

        # Normalization of data (before computing DTW)
        scaler = MinMaxScaler()  # Using MinMaxScaler for normalization
        data_daily_values = scaler.fit_transform(data_daily)

        # Dropdown for choosing linkage method
        linkage_method = st.selectbox("Pilih Metode Linkage", options=["complete", "single", "average"])

        # Compute DTW distance matrix for normalized daily data using fastdtw
        dtw_distance_matrix_daily = compute_dtw_distance_matrix(data_daily_values)

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
        
        plt.title('Silhouette Score vs. Jumlah Kluster (Data Harian)')
        plt.xlabel('Jumlah Kluster')
        plt.ylabel('Silhouette Score')
        plt.xticks(range(2, max_n_clusters + 1))
        plt.grid(True)
        st.pyplot(plt)

        # Determine optimal number of clusters
        optimal_n_clusters = max(silhouette_scores, key=silhouette_scores.get)
        st.write(f"Jumlah kluster optimal berdasarkan Silhouette Score adalah: {optimal_n_clusters}")

        # Clustering and dendrogram for linkage method
        condensed_dtw_distance_matrix = squareform(dtw_distance_matrix_daily)
        Z = linkage(condensed_dtw_distance_matrix, method=linkage_method)

        # Dendrogram hanya untuk linkage methods
        if linkage_method in ["complete", "single", "average"]:
            plt.figure(figsize=(16, 10))
            dendrogram(Z, labels=data_daily.columns, leaf_rotation=90)
            plt.title(f'Dendrogram Clustering dengan DTW (Data Harian) - Linkage: {linkage_method.capitalize()}')
            plt.xlabel('Provinsi')
            plt.ylabel('Jarak DTW')
            st.pyplot(plt)

        # KMedoids clustering
        st.subheader("Clustering dengan KMedoids")
        n_clusters_kmedoids = st.slider("Jumlah Kluster KMedoids", 2, 10, 3)

        kmedoids = KMedoids(n_clusters=n_clusters_kmedoids, metric="precomputed", init="k-medoids++", random_state=42)
        kmedoids_labels = kmedoids.fit_predict(dtw_distance_matrix_daily)

        # Show the KMedoids clustering results (without dendrogram)
        st.write(f"Cluster hasil KMedoids (Jumlah kluster = {n_clusters_kmedoids}):")
        st.write(pd.DataFrame({
            'Province': data_daily.columns,
            'Cluster': kmedoids_labels
        }))

        # GeoJSON mapping for linkage methods only
        if linkage_method in ["complete", "single", "average"]:
            gdf = upload_geojson_file()

            if gdf is not None:
                gdf = gdf.rename(columns={'Propinsi': 'Province'})  # Change according to the correct column name
                gdf['Province'] = gdf['Province'].str.upper().str.replace('.', '', regex=False).str.strip()

                # Calculate cluster from clustering results
                clustered_data = pd.DataFrame({
                    'Province': data_daily.columns,
                    'Cluster': kmedoids_labels
                })

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
                plt.title("Pemetaan Provinsi Berdasarkan Kluster")
                st.pyplot(fig)

# Ensure DTW distance matrix is symmetric
def symmetrize(matrix):
    return (matrix + matrix.T) / 2

# Function to compute DTW distance matrix using fastdtw
def compute_dtw_distance_matrix(data_df: np.array) -> np.array:
    num_provinces = data_df.shape[1]
    dtw_distance_matrix = np.zeros((num_provinces, num_provinces))

    for i in range(num_provinces):
        for j in range(i, num_provinces):
            # Reshape to 1D array for DTW calculation
            dtw_distance_matrix[i, j] = fastdtw(data_df[:, i], data_df[:, j])[0]
            dtw_distance_matrix[j, i] = dtw_distance_matrix[i, j]

    return dtw_distance_matrix

# Main function to run Streamlit application
def main():
    st.sidebar.title("Menu Aplikasi")
    selected_option = option_menu(
        menu_title=None,
        options=["Statistika Deskriptif", "Pemetaan"],
        icons=["bar-chart-line", "map"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal"
    )

    # Upload CSV file
    df = upload_csv_file()

    if selected_option == "Statistika Deskriptif":
        statistika_deskriptif(df)
    elif selected_option == "Pemetaan":
        pemetaan(df)

if __name__ == "__main__":
    main()
