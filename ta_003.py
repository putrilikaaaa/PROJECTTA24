import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import dendrogram, linkage
import geopandas as gpd

# Fungsi untuk standardisasi data
def standardize_data(data):
    return (data - data.mean()) / data.std()

# Fungsi untuk menghitung matriks jarak lokal (local cost matrix)
def compute_local_cost_matrix(data):
    return np.abs(data[:, None] - data[None, :])

# Fungsi untuk menghitung matriks jarak terakumulasi (accumulated cost matrix)
def compute_accumulated_cost_matrix(local_cost_matrix):
    return np.cumsum(local_cost_matrix, axis=1)

# Fungsi untuk menghitung matriks jarak DTW
def compute_dtw_distance_matrix(accumulated_cost_matrix):
    return accumulated_cost_matrix[:, -1]

# Fungsi untuk memastikan matriks jarak simetris
def symmetrize(matrix):
    return (matrix + matrix.T) / 2

# Fungsi untuk mengunggah file GeoJSON
def upload_geojson_file():
    try:
        gdf = gpd.read_file("/path/to/indonesia-prov.geojson")
        return gdf
    except Exception as e:
        st.error(f"Error loading GeoJSON file: {e}")
        return None

# Statistika Deskriptif Page
def statistika_deskriptif(data_df):
    st.subheader("Statistika Deskriptif")

    if data_df is not None:
        data_df['Tanggal'] = pd.to_datetime(data_df['Tanggal'], format='%d-%b-%y', errors='coerce')
        data_df.set_index('Tanggal', inplace=True)

        st.write("Deskripsi Statistik Data:")
        st.write(data_df.describe())

        st.write("Visualisasi Data:")
        selected_province = st.selectbox("Pilih Provinsi", options=data_df.columns)

        plt.figure(figsize=(10, 5))
        plt.plot(data_df.index, data_df[selected_province], marker='o', linestyle='-', color='b')
        plt.title(f"Tren Harga di {selected_province}")
        plt.xlabel("Tanggal")
        plt.ylabel("Harga")
        plt.grid(True)
        st.pyplot(plt)

        # Visualisasi Peta berdasarkan Tanggal
        selected_date = st.date_input("Pilih Tanggal", value=data_df.index.min().date())

        if pd.to_datetime(selected_date) in data_df.index:
            plt.figure(figsize=(10, 5))
            plt.bar(data_df.columns, data_df.loc[pd.to_datetime(selected_date)], color='b')
            plt.title(f"Harga pada Tanggal {selected_date}")
            plt.xlabel("Provinsi")
            plt.ylabel("Harga")
            plt.xticks(rotation=90)
            st.pyplot(plt)
        else:
            st.warning(f"Tanggal {selected_date} tidak ada dalam data.")


# Pemetaan Page
def pemetaan(data_df):
    st.subheader("Pemetaan Clustering dengan DTW")

    if data_df is not None:
        data_df['Tanggal'] = pd.to_datetime(data_df['Tanggal'], format='%d-%b-%y', errors='coerce')
        data_df.set_index('Tanggal', inplace=True)

        # Calculate daily averages
        data_daily = data_df.resample('D').mean()
        data_daily.fillna(method='ffill', inplace=True)

        data_daily = standardize_data(data_daily)

        data_daily_values = data_daily.values

        linkage_method = st.selectbox("Pilih Metode Linkage", options=["complete", "single", "average"])

        local_cost_matrix_daily = compute_local_cost_matrix(data_daily)
        accumulated_cost_matrix_daily = compute_accumulated_cost_matrix(local_cost_matrix_daily)

        dtw_distance_matrix_daily = compute_dtw_distance_matrix(accumulated_cost_matrix_daily)
        dtw_distance_matrix_daily = symmetrize(dtw_distance_matrix_daily)

        num_samples = dtw_distance_matrix_daily.shape[0]
        
        max_n_clusters = 10
        silhouette_scores = {}
        cluster_labels_dict = {}

        for n_clusters in range(2, max_n_clusters + 1):
            clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage=linkage_method)
            labels = clustering.fit_predict(dtw_distance_matrix_daily)

            if len(labels) == num_samples:
                try:
                    score = silhouette_score(dtw_distance_matrix_daily, labels, metric='precomputed')
                    silhouette_scores[n_clusters] = score
                    cluster_labels_dict[n_clusters] = labels
                except ValueError as e:
                    st.error(f"Error calculating silhouette score: {e}")
            else:
                st.error("Jumlah label tidak sesuai dengan ukuran matriks jarak.")

        if silhouette_scores:
            plt.figure(figsize=(10, 6))
            plt.plot(list(silhouette_scores.keys()), list(silhouette_scores.values()), marker='o', linestyle='-')

            for n_clusters, score in silhouette_scores.items():
                plt.text(n_clusters, score, f"{score:.2f}", fontsize=9, ha='right')

            plt.title('Silhouette Score vs. Number of Clusters (Data Harian)')
            plt.xlabel('Number of Clusters')
            plt.ylabel('Silhouette Score')
            plt.xticks(range(2, max_n_clusters + 1))
            plt.grid(True)
            st.pyplot(plt)

        if silhouette_scores:
            optimal_n_clusters = max(silhouette_scores, key=silhouette_scores.get)
            st.write(f"Jumlah kluster optimal berdasarkan Silhouette Score adalah: {optimal_n_clusters}")
        
            condensed_dtw_distance_matrix = squareform(dtw_distance_matrix_daily)
            Z = linkage(condensed_dtw_distance_matrix, method=linkage_method)

            plt.figure(figsize=(16, 10))
            dendrogram(Z, labels=data_daily.columns, leaf_rotation=90)
            plt.title(f'Dendrogram Clustering dengan DTW (Data Harian) - Linkage: {linkage_method.capitalize()}')
            plt.xlabel('Provinsi')
            plt.ylabel('Jarak DTW')
            st.pyplot(plt)

            cluster_labels = cluster_labels_dict[optimal_n_clusters]
            clustered_data = pd.DataFrame({
                'Province': data_daily.columns,
                'Cluster': cluster_labels
            })

            st.subheader("Tabel Provinsi per Cluster")
            st.write(clustered_data)

            gdf = upload_geojson_file()

            if gdf is not None:
                gdf = gdf.rename(columns={'Propinsi': 'Province'})
                gdf['Province'] = gdf['Province'].str.upper().str.replace('.', '', regex=False).str.strip()

                clustered_data['Province'] = clustered_data['Province'].str.upper().str.replace('.', '', regex=False).str.strip()

                gdf['Province'] = gdf['Province'].replace({
                    'DI ACEH': 'ACEH',
                    'KEPULAUAN BANGKA BELITUNG': 'BANGKA BELITUNG',
                    'NUSATENGGARA BARAT': 'NUSA TENGGARA BARAT',
                    'D.I YOGYAKARTA': 'DI YOGYAKARTA',
                    'DAERAH ISTIMEWA YOGYAKARTA': 'DI YOGYAKARTA',
                })

                gdf = gdf[gdf['Province'].notna()]

                gdf = gdf.merge(clustered_data, on='Province', how='left')

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

                grey_provinces = gdf[gdf['color'] == 'grey']['Province'].tolist()
                if grey_provinces:
                    st.subheader("Provinsi yang Tidak Termasuk dalam Kluster:")
                    st.write(grey_provinces)
                else:
                    st.write("Semua provinsi termasuk dalam kluster.")

                fig, ax = plt.subplots(1, 1, figsize=(12, 10))
                gdf.boundary.plot(ax=ax, linewidth=1, color='black')
                gdf.plot(ax=ax, color=gdf['color'], edgecolor='black', alpha=0.7)
                plt.title("Pemetaan Provinsi Berdasarkan Kluster")
                st.pyplot(fig)

# Fungsi utama Streamlit
def main():
    st.title("Aplikasi Statistika Deskriptif dan Pemetaan Kluster dengan DTW")

    data_file = st.file_uploader("Unggah file CSV", type=["csv"])
    
    if data_file:
        data_df = pd.read_csv(data_file)

        # Tampilkan Statistika Deskriptif dan Pemetaan secara bersamaan
        col1, col2 = st.columns(2)

        with col1:
            statistika_deskriptif(data_df)

        with col2:
            pemetaan(data_df)

if __name__ == "__main__":
    main()
