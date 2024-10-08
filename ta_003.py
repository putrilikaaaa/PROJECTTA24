import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from sklearn_extra.cluster import KMedoids  # Import KMedoids
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
        clustering_method = st.selectbox("Pilih Metode Clustering", 
                                        options=["complete", "single", "average", "kmedoids"])

        if clustering_method == "kmedoids":
            # KMedoids clustering
            num_clusters = st.slider("Jumlah Kluster", min_value=2, max_value=10, value=3)
            kmedoids = KMedoids(n_clusters=num_clusters, metric='euclidean')
            kmedoids_labels = kmedoids.fit_predict(data_daily_values.T)
            silhouette_avg = silhouette_score(data_daily_values.T, kmedoids_labels, metric='euclidean')

            # Display silhouette score
            st.write(f"Silhouette Score untuk KMedoids: {silhouette_avg:.2f}")

            # Prepare cluster results
            clustered_data = pd.DataFrame({
                'Province': data_daily.columns,
                'Cluster': kmedoids_labels
            })

        else:
            # Agglomerative Clustering for other methods
            linkage_method = clustering_method

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
            dendrogram(Z, labels=data_daily.columns, leaf_rotation=90)
            plt.title(f'Dendrogram Clustering dengan DTW (Data Harian) - Linkage: {linkage_method.capitalize()}')
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
            gdf.boundary.plot(ax=ax, linewidth=1, color='black')
            gdf.plot(column='color', ax=ax, legend=True, legend_kwds={'bbox_to_anchor': (1, 1)}, color=gdf['color'])
            ax.set_title('Pemetaan Provinsi dengan Warna berdasarkan Kluster')
            st.pyplot(fig)

# Main App Structure
def main():
    # Application title
    st.title("Aplikasi Clustering Clustering")
    st.sidebar.title("Menu")

    # Sidebar options
    page = option_menu(
        menu_title=None,
        options=["Statistika Deskriptif", "Pemetaan"],
        icons=["bar-chart", "map"],
        default_index=0,
        orientation="vertical"
    )

    # File upload
    data = upload_csv_file()

    if page == "Statistika Deskriptif":
        statistika_deskriptif(data)
    elif page == "Pemetaan":
        pemetaan(data)

if __name__ == '__main__':
    main()

