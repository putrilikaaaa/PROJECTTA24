import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

# Fungsi untuk memuat dan memproses data
def load_and_process_data(file_path: str) -> pd.DataFrame:
    # Memuat data dari file CSV
    data_df = pd.read_csv(file_path)
    # Mengubah kolom 'Tanggal' menjadi format datetime
    data_df['Tanggal'] = pd.to_datetime(data_df['Tanggal'], format='%d-%b-%y')
    # Mengatur kolom 'Tanggal' sebagai index
    data_df.set_index('Tanggal', inplace=True)
    # Menghapus kolom non-numerik jika ada
    return data_df.select_dtypes(include=[float, int])

# Fungsi untuk standarisasi data
def standardize_data(data_df: pd.DataFrame) -> pd.DataFrame:
    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(data_df), index=data_df.index, columns=data_df.columns)

# Fungsi untuk menghitung matriks biaya lokal DTW
def compute_local_cost_matrix(data_df: pd.DataFrame) -> np.array:
    num_provinces = data_df.shape[1]
    num_time_points = data_df.shape[0]
    local_cost_matrix = np.zeros((num_time_points, num_provinces, num_provinces))

    for i in range(num_provinces):
        for j in range(num_provinces):
            if i != j:
                cost = np.square(data_df.iloc[:, i] - data_df.iloc[:, j])
                local_cost_matrix[:, i, j] = cost

    return local_cost_matrix

# Fungsi untuk menghitung matriks biaya akumulatif
def compute_accumulated_cost_matrix(local_cost_matrix: np.array) -> np.array:
    num_time_points, num_provinces, _ = local_cost_matrix.shape
    accumulated_cost_matrix = np.full((num_time_points, num_provinces, num_provinces), np.inf)
    accumulated_cost_matrix[0, :, :] = local_cost_matrix[0, :, :]

    for t in range(1, num_time_points):
        for i in range(num_provinces):
            for j in range(num_provinces):
                min_prev_cost = np.min([
                    accumulated_cost_matrix[t-1, i, j],
                    accumulated_cost_matrix[t-1, i, :].min(),
                    accumulated_cost_matrix[t-1, :, j].min()
                ])
                accumulated_cost_matrix[t, i, j] = local_cost_matrix[t, i, j] + min_prev_cost

    return accumulated_cost_matrix

# Fungsi untuk menghitung jarak DTW
def compute_dtw_distance_matrix(accumulated_cost_matrix: np.array) -> np.array:
    num_provinces = accumulated_cost_matrix.shape[1]
    dtw_distance_matrix = np.zeros((num_provinces, num_provinces))

    for i in range(num_provinces):
        for j in range(i + 1, num_provinces):
            dtw_distance = accumulated_cost_matrix[-1, i, j]
            dtw_distance_matrix[i, j] = dtw_distance
            dtw_distance_matrix[j, i] = dtw_distance

    return dtw_distance_matrix

# Streamlit UI
st.title("Pemetaan Clustering dengan DTW")

# Upload file
uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

if uploaded_file is not None:
    data_df = load_and_process_data(uploaded_file)

    # Menghitung rata-rata per minggu dan bulan
    data_weekly = data_df.resample('W').mean()
    data_monthly = data_df.resample('M').mean()

    # Standarisasi data mingguan dan bulanan
    data_weekly_standardized = standardize_data(data_weekly)
    data_monthly_standardized = standardize_data(data_monthly)

    # Hitung matriks biaya lokal dan akumulatif untuk data mingguan dan bulanan
    local_cost_matrix_weekly = compute_local_cost_matrix(data_weekly_standardized)
    accumulated_cost_matrix_weekly = compute_accumulated_cost_matrix(local_cost_matrix_weekly)

    local_cost_matrix_monthly = compute_local_cost_matrix(data_monthly_standardized)
    accumulated_cost_matrix_monthly = compute_accumulated_cost_matrix(local_cost_matrix_monthly)

    # Hitung matriks jarak DTW untuk data mingguan dan bulanan
    dtw_distance_matrix_weekly = compute_dtw_distance_matrix(accumulated_cost_matrix_weekly)
    dtw_distance_matrix_monthly = compute_dtw_distance_matrix(accumulated_cost_matrix_monthly)

    # Klustering dan perhitungan skor siluet untuk data bulanan
    max_n_clusters = 10
    silhouette_scores = {}
    
    for n_clusters in range(2, max_n_clusters + 1):
        clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='complete')
        labels = clustering.fit_predict(dtw_distance_matrix_monthly)
        score = silhouette_score(dtw_distance_matrix_monthly, labels, metric='precomputed')
        silhouette_scores[n_clusters] = score

    # Plot Silhouette Scores
    plt.figure(figsize=(10, 6))
    plt.plot(list(silhouette_scores.keys()), list(silhouette_scores.values()), marker='o', linestyle='-')
    plt.title('Silhouette Score vs. Number of Clusters (Data Bulanan)')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.xticks(range(2, max_n_clusters + 1))
    plt.grid(True)
    st.pyplot(plt)

    # Klustering dan dendrogram
    condensed_dtw_distance_matrix = squareform(dtw_distance_matrix_monthly)
    Z = linkage(condensed_dtw_distance_matrix, method='complete')

    plt.figure(figsize=(16, 10))
    dendrogram(Z, labels=data_monthly_standardized.columns, leaf_rotation=90)
    plt.title('Dendrogram Clustering dengan DTW (Data Bulanan)')
    plt.xlabel('Provinsi')
    plt.ylabel('Jarak DTW')
    st.pyplot(plt)
