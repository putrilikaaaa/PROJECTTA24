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
from streamlit_option_menu import option_menu
import requests

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
        provinces = [col for col in data_df.columns if col != 'Tanggal']
        province = st.selectbox("Pilih Provinsi", options=provinces)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(data_df.index, data_df[province], label=province)

        min_value = data_df[province].min()
        max_value = data_df[province].max()
        ax.set_ylim(bottom=min_value - (0.1 * abs(max_value - min_value)))

        ax.set_title(f"Tren Data Provinsi {province}")
        ax.set_xlabel("Tanggal")
        ax.set_ylabel("Nilai")
        ax.legend()
        st.line_chart(data_df[province])
        st.write(f"Statistika Deskriptif untuk Provinsi {province}:")
        st.write(data_df[province].describe())

# Pemetaan Page
def pemetaan(data_df):
    st.subheader("Pemetaan dengan Metode Linkage")

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

        # Adjust cluster labels to start from 1 instead of 0
        cluster_labels = cluster_labels_dict[optimal_n_clusters] + 1
        clustered_data = pd.DataFrame({
            'Province': data_daily.columns,
            'Cluster': cluster_labels
        })

        st.subheader("Tabel Label Cluster Setiap Provinsi")
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
                1: 'red',
                2: 'yellow',
                3: 'green',
                4: 'blue',
                5: 'purple',
                6: 'orange',
                7: 'pink',
                8: 'brown',
                9: 'cyan',
                10: 'magenta'
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
    st.subheader("Pemetaan dengan Metode K-Medoids")

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

        plt.title('Silhouette Score vs. Jumlah Kluster (KMedoids)')
        plt.xlabel('Jumlah Kluster')
        plt.ylabel('Silhouette Score')
        plt.xticks(range(2, max_n_clusters + 1))
        plt.grid(True)
        st.pyplot(plt)

        optimal_n_clusters = max(silhouette_scores, key=silhouette_scores.get)
        st.write(f"Jumlah kluster optimal berdasarkan Silhouette Score adalah: {optimal_n_clusters}")

        cluster_labels = cluster_labels_dict[optimal_n_clusters] + 1
        clustered_data = pd.DataFrame({
            'Province': data_daily.columns,
            'Cluster': cluster_labels
        })

        st.subheader("Tabel Label Cluster Setiap Provinsi")
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
                1: 'red',
                2: 'yellow',
                3: 'green',
                4: 'blue',
                5: 'purple',
                6: 'orange',
                7: 'pink',
                8: 'brown',
                9: 'cyan',
                10: 'magenta'
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

# Main function
def download_template():
    # URL to the raw CSV file
    template_url = "https://raw.githubusercontent.com/putrilikaaaa/PROJECTTA24/main/TEMPLATE.csv"
    
    # Fetch the CSV file content
    response = requests.get(template_url)
    response.raise_for_status()  # Raise an error for bad responses
    
    # Return the CSV content
    return response.content

def main():
    st.set_page_config(page_title="Dashboard Clustering", page_icon="📊", layout="wide")

    # Create a download button for the CSV template
    csv_content = download_template()
    st.download_button(
        label="Download CSV Template
        data=csv_content,
        file_name="TEMPLATE.csv",
        mime="text/csv",
    )

    # Allow users to upload data
    st.markdown("## Upload Data")
    data_df = upload_csv_file()

    # Create a sidebar menu for navigation
    with st.sidebar:
        selected = option_menu("Menu", ["Statistika Deskriptif", "Pemetaan Linkage", "Pemetaan KMedoids"],
                               icons=['bar-chart', 'map', 'map'], menu_icon="cast", default_index=0)

    # Load the appropriate page based on user selection
    if selected == "Statistika Deskriptif":
        statistika_deskriptif(data_df)
    elif selected == "Pemetaan Linkage":
        pemetaan(data_df)
    elif selected == "Pemetaan KMedoids":
        pemetaan_kmedoids(data_df)

if __name__ == "__main__":
    main()
