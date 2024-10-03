import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import silhouette_score

# Fungsi untuk menghitung matriks biaya lokal
def compute_local_cost_matrix(data):
    return np.abs(data[:, None] - data[None, :])

# Fungsi untuk menghitung matriks biaya terakumulasi
def compute_accumulated_cost_matrix(local_cost_matrix):
    n = local_cost_matrix.shape[0]
    accumulated_cost = np.zeros((n, n))

    # Pengecekan ukuran matriks lokal
    if n == 0 or local_cost_matrix.shape[1] != n:
        raise ValueError("Matriks biaya lokal tidak valid.")

    for i in range(n):
        for j in range(n):
            if i == 0 and j == 0:
                accumulated_cost[i, j] = local_cost_matrix[i, j]
            elif i == 0:
                accumulated_cost[i, j] = accumulated_cost[i, j - 1] + local_cost_matrix[i, j]
            elif j == 0:
                accumulated_cost[i, j] = accumulated_cost[i - 1, j] + local_cost_matrix[i, j]
            else:
                accumulated_cost[i, j] = min(accumulated_cost[i - 1, j], accumulated_cost[i, j - 1]) + local_cost_matrix[i, j]
    return accumulated_cost

# Fungsi untuk menghitung jarak DTW
def compute_dtw_distance_matrix(accumulated_cost_matrix):
    return accumulated_cost_matrix[-1, -1]

# Fungsi untuk menormalkan data
def standardize_data(data):
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)

# Fungsi untuk menyimetrisasi matriks
def symmetrize(matrix):
    return (matrix + matrix.T) / 2

# Fungsi untuk halaman pemetaan
def pemetaan(data_df):
    st.subheader("Pemetaan Clustering dengan DTW")

    if data_df is not None:
        if 'Tanggal' in data_df.columns:
            data_df['Tanggal'] = pd.to_datetime(data_df['Tanggal'], format='%d-%b-%y', errors='coerce')
            data_df.set_index('Tanggal', inplace=True)

            # Menghitung rata-rata harian
            data_daily = data_df.resample('D').mean()
            data_daily.fillna(method='ffill', inplace=True)

            # Memastikan ada data untuk pemrosesan
            if data_daily.empty:
                st.error("Data harian kosong setelah resampling.")
                return

            # Standardisasi data
            data_daily_values = standardize_data(data_daily.values)

            # Menghitung matriks biaya lokal
            local_cost_matrix_daily = compute_local_cost_matrix(data_daily_values)

            # Memeriksa ukuran matriks biaya lokal
            if local_cost_matrix_daily.size == 0 or local_cost_matrix_daily.shape[0] != local_cost_matrix_daily.shape[1]:
                st.error("Matriks biaya lokal tidak valid. Pastikan data input Anda benar.")
                return

            # Menghitung matriks biaya terakumulasi
            accumulated_cost_matrix_daily = compute_accumulated_cost_matrix(local_cost_matrix_daily)

            # Menghitung jarak DTW
            dtw_distance_matrix_daily = compute_dtw_distance_matrix(accumulated_cost_matrix_daily)

            # Memastikan dtw_distance_matrix_daily adalah array 2D sebelum disimetrisasi
            dtw_distance_matrix_daily = symmetrize(np.expand_dims(dtw_distance_matrix_daily, axis=0))

            # Memastikan kita memiliki label untuk silhouette score
            # Ganti dengan logika yang tepat untuk mendapatkan labels
            labels = ...  # Pastikan Anda memiliki label untuk clustering
            if len(labels) != dtw_distance_matrix_daily.shape[0]:
                st.error("Jumlah label tidak cocok dengan jumlah data.")
                return

            # Menghitung silhouette score
            score = silhouette_score(dtw_distance_matrix_daily, labels, metric='precomputed')

            st.write(f"Silhouette Score: {score}")

# Fungsi utama untuk menjalankan Streamlit
def main():
    st.title("Aplikasi Pemodelan dan Pemetaan Data")
    
    uploaded_file = st.file_uploader("Unggah File Data", type=["csv", "xlsx"])
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            data_df = pd.read_csv(uploaded_file)
        else:
            data_df = pd.read_excel(uploaded_file)

        st.write(data_df.head())
        
        # Memanggil fungsi pemetaan
        pemetaan(data_df)

if __name__ == "__main__":
    main()
