import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from sklearn_extra.cluster import KMedoids  # Import KMedoids from sklearn_extra
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
import geopandas as gpd

# Function to upload CSV files
def upload_csv_file(key=None):
    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"], key=key)
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

# Descriptive Statistics Page
def statistik_deskriptif():
    st.subheader("Statistika Deskriptif")
    data_df = upload_csv_file()

    if data_df is not None:
        st.write("Dataframe:")
        st.write(data_df)

        # Convert 'Tanggal' column to datetime and set as index
        data_df['Tanggal'] = pd.to_datetime(data_df['Tanggal'], format='%d-%b-%y', errors='coerce')
        data_df.set_index('Tanggal', inplace=True)

        if data_df.isnull().any().any():
            st.warning("Terdapat tanggal yang tidak valid, silakan periksa data Anda.")

        # Dropdown for selecting province
        selected_province = st.selectbox("Pilih Provinsi", options=data_df.columns.tolist())

        if selected_province:
            # Display descriptive statistics
            st.subheader(f"Statistika Deskriptif untuk {selected_province}")
            st.write(data_df[selected_province].describe())

            # Display line chart
            st.subheader(f"Line Chart untuk {selected_province}")
            plt.figure(figsize=(12, 6))
            plt.plot(data_df.index, data_df[selected_province], label=selected_province, color='blue')
            plt.title(f"Line Chart - {selected_province}")
            plt.xlabel('Tanggal')
            plt.ylabel('Nilai')
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(plt)

# Mapping Page
def pemetaan():
    st.subheader("Pemetaan Clustering dengan DTW")
    data_df = upload_csv_file(key="pemetaan_upload")

    if data_df is not None:
        data_df['Tanggal'] = pd.to_datetime(data_df['Tanggal'], format='%d-%b-%y', errors='coerce')
        data_df.set_index('Tanggal', inplace=True)

        # Calculate daily averages
        data_daily = data_df.resample('D').mean()

        # Standardize daily data
        scaler = StandardScaler()
        data_daily_standardized = pd.DataFrame(scaler.fit_transform(data_daily), index=data_daily.index, columns=data_daily.columns)

        # Compute local cost matrix and accumulated cost matrix
        local_cost_matrix_daily = compute_local_cost_matrix(data_daily_standardized)
        accumulated_cost_matrix_daily = compute_accumulated_cost_matrix(local_cost_matrix_daily)

        # Compute DTW distance matrix for daily data
        dtw_distance_matrix_daily = compute_dtw_distance_matrix(accumulated_cost_matrix_daily)

        # Dropdown for selecting clustering method
        clustering_method = st.selectbox("Pilih Metode Clustering", ["Complete Linkage", "K-Medoids"])

        if clustering_method == "Complete Linkage":
            # Clustering and silhouette score calculation for complete linkage
            max_n_clusters = 10
            silhouette_scores = {}

            for n_clusters in range(2, max_n_clusters + 1):
                clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='complete')
                labels = clustering.fit_predict(dtw_distance_matrix_daily)
                score = silhouette_score(dtw_distance_matrix_daily, labels, metric='precomputed')
                silhouette_scores[n_clusters] = score

            # Plot Silhouette Scores
            plt.figure(figsize=(10, 6))
            plt.plot(list(silhouette_scores.keys()), list(silhouette_scores.values()), marker='o', linestyle='-')
            plt.title('Silhouette Score vs. Number of Clusters (Complete Linkage)')
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
            Z = linkage(condensed_dtw_distance_matrix, method='complete')

            plt.figure(figsize=(16, 10))
            dendrogram(Z, labels=data_daily_standardized.columns, leaf_rotation=90)
            plt.title('Dendrogram Clustering dengan DTW (Complete Linkage)')
            plt.xlabel('Provinsi')
            plt.ylabel('Jarak DTW')
            st.pyplot(plt)

            # Clustering results
            cluster_labels = AgglomerativeClustering(n_clusters=optimal_n_clusters, metric='precomputed', linkage='complete').fit_predict(dtw_distance_matrix_daily)
        
        elif clustering_method == "K-Medoids":
            # Calculate silhouette scores for K-Medoids
            max_n_clusters = 10
            silhouette_scores = {}

            for n_clusters in range(2, max_n_clusters + 1):
                clustering = KMedoids(n_clusters=n_clusters, metric='precomputed')
                labels = clustering.fit_predict(dtw_distance_matrix_daily)
                score = silhouette_score(dtw_distance_matrix_daily, labels, metric='precomputed')
                silhouette_scores[n_clusters] = score

            # Plot Silhouette Scores
            plt.figure(figsize=(10, 6))
            plt.plot(list(silhouette_scores.keys()), list(silhouette_scores.values()), marker='o', linestyle='-')
            plt.title('Silhouette Score vs. Number of Clusters (K-Medoids)')
            plt.xlabel('Number of Clusters')
            plt.ylabel('Silhouette Score')
            plt.xticks(range(2, max_n_clusters + 1))
            plt.grid(True)
            st.pyplot(plt)

            # Determine optimal number of clusters
            optimal_n_clusters = max(silhouette_scores, key=silhouette_scores.get)
            st.write(f"Jumlah kluster optimal berdasarkan Silhouette Score adalah: {optimal_n_clusters}")

            # Clustering results
            clustering = KMedoids(n_clusters=optimal_n_clusters, metric='precomputed')
            cluster_labels = clustering.fit_predict(dtw_distance_matrix_daily)

        # Create DataFrame for clustered results
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
                2: 'green'
            })
            gdf['color'].fillna('grey', inplace=True)

            # Display provinces colored grey
            grey_provinces = gdf[gdf['color'] == 'grey']['Province'].tolist()
            if grey_provinces:
                st.subheader("Provinsi yang Tidak Termasuk dalam Kluster:")
                st.write(grey_provinces)
            else:
                st.subheader("Semua provinsi termasuk dalam kluster.")

            # Plotting the map
            st.subheader("Pemetaan Kluster")
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            gdf.boundary.plot(ax=ax, linewidth=1)
            gdf.plot(column='color', ax=ax, legend=True)
            plt.title('Pemetaan Kluster Provinsi')
            plt.axis('off')
            st.pyplot(fig)

# Main function to control the app
def main():
    st.title("Aplikasi Analisis Data dan Clustering")

    # Sidebar for navigation
    pages = {
        "Statistika Deskriptif": statistik_deskriptif,
        "Pemetaan": pemetaan
    }
    page = st.sidebar.selectbox("Pilih Halaman", options=list(pages.keys()))

    # Call the selected page function
    pages[page]()

if __name__ == "__main__":
    main()
