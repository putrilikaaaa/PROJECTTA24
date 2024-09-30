import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform
import geopandas as gpd

# Fungsi untuk menghitung DTW distance matrix
def calculate_dtw_distance(data):
    # Implementasikan logika untuk menghitung DTW distance matrix
    # Misalnya, menggunakan pdist dan squareform dari scipy
    distance_matrix = squareform(pdist(data, metric='euclidean'))  # Ganti dengan DTW jika perlu
    return distance_matrix

# Fungsi untuk pemetaan
def pemetaan():
    # Load data
    # Misalnya, ganti dengan cara Anda memuat data
    data_daily = pd.read_csv('path_to_your_daily_data.csv')

    # Hitung DTW distance matrix
    dtw_distance_matrix_daily = calculate_dtw_distance(data_daily)

    st.write("Shape of DTW distance matrix:", dtw_distance_matrix_daily.shape)
    st.write("Data type of DTW distance matrix:", dtw_distance_matrix_daily.dtype)

    if dtw_distance_matrix_daily.shape[0] < 2 or dtw_distance_matrix_daily.shape[1] < 2:
        st.error("The DTW distance matrix must have at least two samples.")
        return

    if np.isnan(dtw_distance_matrix_daily).any() or np.isinf(dtw_distance_matrix_daily).any():
        st.error("The DTW distance matrix contains NaN or infinity values.")
        return

    if not np.allclose(dtw_distance_matrix_daily, dtw_distance_matrix_daily.T):
        st.error("The DTW distance matrix is not symmetric.")
        return

    dtw_distance_matrix_daily = dtw_distance_matrix_daily.astype(np.float64)

    # Clustering
    clustering = AgglomerativeClustering(n_clusters=3)  # Ubah sesuai kebutuhan Anda
    try:
        labels = clustering.fit_predict(dtw_distance_matrix_daily)
    except ValueError as e:
        st.error(f"Clustering error: {e}")
        return

    # Visualisasi hasil clustering
    data_daily['Cluster'] = labels
    st.write("Cluster assignments:")
    st.write(data_daily)

    # Pemetaan menggunakan GeoDataFrame
    gdf = gpd.read_file('path_to_your_geojson.geojson')  # Ganti dengan path GeoJSON Anda

    # Gabungkan data dengan GeoDataFrame
    gdf = gdf.merge(data_daily, how='left', left_on='province', right_on='province')

    # Visualisasi peta
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    gdf.boundary.plot(ax=ax, linewidth=1)
    gdf.plot(column='Cluster', cmap='RdYlGn', legend=True, ax=ax)
    st.pyplot(fig)

# Fungsi utama untuk menjalankan aplikasi
def main():
    st.title("Aplikasi Pemetaan Clustering")
    
    # Tambahkan menu atau komponen lain jika diperlukan
    menu = st.sidebar.selectbox("Pilih Menu", ["Pemetaan"])
    
    if menu == "Pemetaan":
        pemetaan()

if __name__ == "__main__":
    main()
