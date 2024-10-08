import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
import geopandas as gpd
from fastdtw import fastdtw  # Mengimpor fastdtw untuk perhitungan DTW

# Fungsi untuk mengunggah file CSV
def upload_csv_file():
    uploaded_file = st.file_uploader("Unggah File CSV", type=['csv'])
    if uploaded_file is not None:
        data_df = pd.read_csv(uploaded_file)
        st.write("Data yang diunggah:")
        st.write(data_df)
        return data_df
    return None

# Fungsi untuk mengunggah file GeoJSON
def upload_geojson_file():
    try:
        gdf = gpd.read_file('https://raw.githubusercontent.com/username/repo/branch/indonesia-prov.geojson')
        return gdf
    except Exception as e:
        st.error(f"Terjadi kesalahan dalam mengunggah GeoJSON: {e}")
        return None

# Fungsi untuk menghitung matriks biaya lokal
def compute_local_cost_matrix(data_df):
    n = len(data_df.columns)
    local_cost_matrix = pd.DataFrame(index=data_df.columns, columns=data_df.columns)

    for i in range(n):
        for j in range(n):
            # Menghitung biaya DTW antara dua seri data menggunakan fastdtw
            local_cost_matrix.iloc[i, j] = fastdtw(data_df.iloc[:, i], data_df.iloc[:, j])[0]

    return local_cost_matrix

# Fungsi untuk menghitung matriks biaya yang terakumulasi
def compute_accumulated_cost_matrix(local_cost_matrix):
    return local_cost_matrix.cumsum(axis=1).cumsum(axis=0)

# Fungsi untuk menghitung matriks jarak DTW
def compute_dtw_distance_matrix(accumulated_cost_matrix):
    return accumulated_cost_matrix.iloc[-1, -1]

# Fungsi utama untuk Pemetaan
def pemetaan(data_df):
    st.subheader("Pemetaan Clustering dengan DTW")

    if data_df is not None:
        # Inspect columns to debug
        st.write("Columns in the uploaded data:")
        st.write(data_df.columns)

        # Ensure 'Tanggal' column exists
        if 'Tanggal' in data_df.columns:
            # Convert 'Tanggal' column to datetime format
            data_df['Tanggal'] = pd.to_datetime(data_df['Tanggal'], format='%d-%b-%y', errors='coerce')
            data_df.set_index('Tanggal', inplace=True)
        else:
            st.error("Kolom 'Tanggal' tidak ditemukan dalam data.")
            return

        # Calculate daily averages
        data_daily = data_df.resample('D').mean()

        # Handle missing data by forward filling
        data_daily.fillna(method='ffill', inplace=True)

        # Standardization of data
        scaler = StandardScaler()
        data_daily_values = scaler.fit_transform(data_daily)

        # Dropdown for choosing linkage method
        linkage_method = st.selectbox("Pilih Metode Linkage", options=["complete", "single", "average"])

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
            
            # Ensure labels are in the correct format for silhouette score
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
            gdf.plot(ax=ax, color=gdf['color'])
            st.pyplot(fig)

# Main Function
def main():
    st.title("Clustering dan Visualisasi dengan DTW")
    
    # File upload page
    data_df = upload_csv_file()

    if data_df is not None:
        pemetaan(data_df)

if __name__ == "__main__":
    main()
