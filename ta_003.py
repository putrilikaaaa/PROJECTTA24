import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from fastdtw import fastdtw

# Fungsi untuk menghitung matriks biaya lokal
def compute_local_cost_matrix(data):
    n_series = data.shape[1]
    local_cost_matrix = np.zeros((n_series, n_series))

    for i in range(n_series):
        for j in range(n_series):
            # Hitung jarak DTW untuk setiap pasangan seri
            if i != j:
                local_cost_matrix[i, j] = fastdtw(data.iloc[:, i].values, data.iloc[:, j].values)[0]
            else:
                local_cost_matrix[i, j] = 0  # Set jarak diri ke 0

    return local_cost_matrix

# Fungsi untuk pemetaan
def pemetaan(data_df):
    st.subheader("Pemetaan Clustering dengan DTW")

    if data_df is not None:
        data_df['Tanggal'] = pd.to_datetime(data_df['Tanggal'], format='%d-%b-%y', errors='coerce')
        data_df.set_index('Tanggal', inplace=True)

        # Hitung rata-rata harian
        data_daily = data_df.resample('D').mean()

        # Standarisasi data harian
        scaler = StandardScaler()
        data_daily_standardized = pd.DataFrame(scaler.fit_transform(data_daily), index=data_daily.index, columns=data_daily.columns)

        # Dropdown untuk memilih metode linkage
        linkage_method = st.selectbox("Pilih Metode Linkage", options=["complete", "single", "average"])

        # Hitung matriks biaya lokal dan matriks jarak DTW
        local_cost_matrix_daily = compute_local_cost_matrix(data_daily_standardized)

        # Pastikan matriks biaya lokal adalah persegi dan valid
        if local_cost_matrix_daily.shape[0] == local_cost_matrix_daily.shape[1]:
            # Periksa apakah matriks tidak kosong
            if not np.any(np.isnan(local_cost_matrix_daily)):
                dtw_distance_matrix_daily = squareform(local_cost_matrix_daily)

                # Pastikan matriks jarak DTW simetris
                dtw_distance_matrix_daily = (dtw_distance_matrix_daily + dtw_distance_matrix_daily.T) / 2

                # Clustering dan perhitungan silhouette score untuk data harian
                max_n_clusters = 10
                silhouette_scores = {}

                for n_clusters in range(2, max_n_clusters + 1):
                    clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage=linkage_method)
                    labels = clustering.fit_predict(dtw_distance_matrix_daily)

                    # Hitung silhouette score
                    if len(set(labels)) > 1:  # Pastikan ada lebih dari satu kluster
                        score = silhouette_score(dtw_distance_matrix_daily, labels, metric='precomputed')
                        silhouette_scores[n_clusters] = score

                # Plot Silhouette Scores
                plt.figure(figsize=(10, 6))
                plt.plot(list(silhouette_scores.keys()), list(silhouette_scores.values()), marker='o', linestyle='-')
                
                # Tambahkan label data pada plot silhouette score
                for n_clusters, score in silhouette_scores.items():
                    plt.text(n_clusters, score, f"{score:.2f}", fontsize=9, ha='right')
                
                plt.title('Silhouette Score vs. Number of Clusters (Data Harian)')
                plt.xlabel('Number of Clusters')
                plt.ylabel('Silhouette Score')
                plt.xticks(range(2, max_n_clusters + 1))
                plt.grid(True)
                st.pyplot(plt)

                # Tentukan jumlah kluster optimal
                if silhouette_scores:
                    optimal_n_clusters = max(silhouette_scores, key=silhouette_scores.get)
                    st.write(f"Jumlah kluster optimal berdasarkan Silhouette Score adalah: {optimal_n_clusters}")

                    # Clustering dan dendrogram
                    condensed_dtw_distance_matrix = squareform(dtw_distance_matrix_daily)
                    Z = linkage(condensed_dtw_distance_matrix, method=linkage_method)

                    plt.figure(figsize=(16, 10))
                    dendrogram(Z, labels=data_daily_standardized.columns, leaf_rotation=90)
                    plt.title(f'Dendrogram Clustering dengan DTW (Data Harian) - Linkage: {linkage_method.capitalize()}')
                    plt.xlabel('Provinsi')
                    plt.ylabel('Jarak DTW')
                    st.pyplot(plt)

                    # Tabel provinsi per kluster
                    cluster_labels = AgglomerativeClustering(n_clusters=optimal_n_clusters, metric='precomputed', linkage=linkage_method).fit_predict(dtw_distance_matrix_daily)
                    clustered_data = pd.DataFrame({
                        'Province': data_daily_standardized.columns,
                        'Cluster': cluster_labels
                    })

                    # Tampilkan tabel kluster
                    st.subheader("Tabel Provinsi per Cluster")
                    st.write(clustered_data)

                    # Muat file GeoJSON dari GitHub
                    gdf = upload_geojson_file()

                    if gdf is not None:
                        gdf = gdf.rename(columns={'Propinsi': 'Province'})  # Sesuaikan dengan nama kolom yang benar
                        gdf['Province'] = gdf['Province'].str.upper().str.replace('.', '', regex=False).str.strip()

                        # Hitung kluster dari hasil clustering
                        clustered_data['Province'] = clustered_data['Province'].str.upper().str.replace('.', '', regex=False).str.strip()

                        # Ganti nama provinsi yang tidak konsisten
                        gdf['Province'] = gdf['Province'].replace({
                            'DI ACEH': 'ACEH',
                            'KEPULAUAN BANGKA BELITUNG': 'BANGKA BELITUNG',
                            'NUSATENGGARA BARAT': 'NUSA TENGGARA BARAT',
                            'D.I YOGYAKARTA': 'DI YOGYAKARTA',
                            'DAERAH ISTIMEWA YOGYAKARTA': 'DI YOGYAKARTA',
                        })

                        # Hapus provinsi yang None (yaitu, GORONTALO)
                        gdf = gdf[gdf['Province'].notna()]

                        # Gabungkan data yang dikelompokkan dengan GeoDataFrame
                        gdf = gdf.merge(clustered_data, on='Province', how='left')

                        # Set warna untuk kluster
                        gdf['color'] = gdf['Cluster'].map({
                            0: 'red',
                            1: 'yellow',
                            2: 'green'
                        })
                        gdf['color'].fillna('grey', inplace=True)

                        # Tampilkan provinsi berwarna abu-abu
                        grey_provinces = gdf[gdf['color'] == 'grey']['Province'].tolist()
                        if grey_provinces:
                            st.subheader("Provinsi yang Tidak Termasuk dalam Kluster:")
                            st.write(grey_provinces)
                        else:
                            st.write("Semua provinsi termasuk dalam kluster.")

                        # Plot peta
                        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
                        gdf.boundary.plot(ax=ax, linewidth=1, color='black')  # Plot batas
                        gdf.plot(ax=ax, color=gdf['color'], edgecolor='black', alpha=0.6)  # Plot provinsi dengan warna

                        # Tambahkan judul dan label
                        plt.title('Peta Kluster Provinsi di Indonesia', fontsize=15)
                        plt.xlabel('Longitude', fontsize=12)
                        plt.ylabel('Latitude', fontsize=12)
                        st.pyplot(plt)
                    else:
                        st.warning("Silakan upload file GeoJSON dengan benar.")
                else:
                    st.warning("Tidak ada silhouette scores yang dihitung.")
            else:
                st.error("Matriks biaya lokal mengandung nilai NaN. Pastikan data tidak kosong.")
        else:
            st.error("Matriks biaya lokal tidak valid. Pastikan data memiliki kolom yang benar.")

# Fungsi utama
def main():
    st.title("Aplikasi Clustering Provinsi Indonesia")
    
    uploaded_file = st.file_uploader("Upload File Data", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        if uploaded_file.type == "text/csv":
            data_df = pd.read_csv(uploaded_file)
        else:
            data_df = pd.read_excel(uploaded_file)

        pemetaan(data_df)

if __name__ == "__main__":
    main()
