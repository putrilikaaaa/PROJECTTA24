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
from sklearn.preprocessing import MinMaxScaler
from fastdtw import fastdtw

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
        # Exclude 'Tanggal' from dropdown
        province_columns = [col for col in data_df.columns if col != 'Tanggal']
        province = st.selectbox("Pilih Provinsi", options=province_columns)

        # Display line chart for the selected province
        st.line_chart(data_df[province])

        # Show descriptive statistics for the selected province
        st.write(f"Statistika Deskriptif untuk Provinsi {province}:")
        st.write(data_df[province].describe())

# Pemetaan Page (with single, complete, and average linkage)
def pemetaan(data_df):
    st.subheader("Pemetaan Clustering dengan DTW")

    if data_df is not None:
        data_df['Tanggal'] = pd.to_datetime(data_df['Tanggal'], format='%d-%b-%y', errors='coerce')
        data_df.set_index('Tanggal', inplace=True)

        data_daily = data_df.resample('D').mean()
        data_daily.fillna(method='ffill', inplace=True)

        scaler = MinMaxScaler()
        data_daily_values = scaler.fit_transform(data_daily)

        linkage_method = st.selectbox("Pilih Metode Linkage", options=["complete", "single", "average"])

        dtw_distance_matrix_daily = compute_dtw_distance_matrix(data_daily_values)
        dtw_distance_matrix_daily = symmetrize(dtw_distance_matrix_daily)

        max_n_clusters = 10
        silhouette_scores = {}
        cluster_labels_dict = {}

        for n_clusters in range(2, max_n_clusters + 1):
            clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage=linkage_method)
            labels = clustering.fit_predict(dtw_distance_matrix_daily)
            score = silhouette_score(dtw_distance_matrix_daily, labels, metric='precomputed')
            silhouette_scores[n_clusters] = score
            cluster_labels_dict[n_clusters] = labels

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

        # Peta for Pemetaan
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
            plt.title(f"Pemetaan Provinsi per Kluster - Agglomerative (DTW)")
            st.pyplot(fig)

# Pemetaan KMedoids Page
def pemetaan_kmedoids(data_df):
    st.subheader("Pemetaan KMedoids")

    if data_df is not None:
        data_df['Tanggal'] = pd.to_datetime(data_df['Tanggal'], format='%d-%b-%y', errors='coerce')
        data_df.set_index('Tanggal', inplace=True)

        data_daily = data_df.resample('D').mean()
        data_daily.fillna(method='ffill', inplace=True)

        scaler = MinMaxScaler()
        data_daily_values = scaler.fit_transform(data_daily)

        max_n_clusters = 10
        silhouette_scores = {}
        cluster_labels_dict = {}

        for n_clusters in range(2, max_n_clusters + 1):
            kmedoids = KMedoids(n_clusters=n_clusters, metric="euclidean", random_state=42)
            labels = kmedoids.fit_predict(data_daily_values.T)
            score = silhouette_score(data_daily_values.T, labels)
            silhouette_scores[n_clusters] = score
            cluster_labels_dict[n_clusters] = labels

        plt.figure(figsize=(10, 6))
        plt.plot(list(silhouette_scores.keys()), list(silhouette_scores.values()), marker='o', linestyle='-')
        
        for n_clusters, score in silhouette_scores.items():
            plt.text(n_clusters, score, f"{score:.2f}", fontsize=9, ha='right')
        
        plt.title('Silhouette Score vs. Number of Clusters (KMedoids)')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')
        plt.xticks(range(2, max_n_clusters + 1))
        plt.grid(True)
        st.pyplot(plt)

        optimal_n_clusters = max(silhouette_scores, key=silhouette_scores.get)
        st.write(f"Jumlah kluster optimal berdasarkan Silhouette Score adalah: {optimal_n_clusters}")

        cluster_labels = cluster_labels_dict[optimal_n_clusters]
        clustered_data = pd.DataFrame({
            'Province': data_daily.columns,
            'Cluster': cluster_labels
        })

        st.subheader("Tabel Provinsi per Cluster")
        st.write(clustered_data)

        # Peta for Pemetaan KMedoids
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
            plt.title(f"Pemetaan Provinsi per Kluster - KMedoids")
            st.pyplot(fig)

# Main app logic
def main():
    st.title("Analisis Data Provinsi")
    data_df = upload_csv_file()

    if data_df is not None:
        statistika_deskriptif(data_df)
        pemetaan(data_df)
        pemetaan_kmedoids(data_df)

if __name__ == "__main__":
    main()
