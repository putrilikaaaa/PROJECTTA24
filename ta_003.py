import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

# Fungsi untuk mengupload file
def upload_file():
    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            return df
        except Exception as e:
            st.error(f"Error: {e}")
            return None
    else:
        return None

# Halaman Statistika Deskriptif
def statistik_deskriptif():
    st.subheader("Statistika Deskriptif")
    data_df = upload_file()

    if data_df is not None:
        st.write("Dataframe:")
        st.write(data_df)
        
        # Mengubah kolom 'Tanggal' menjadi format datetime dan mengatur sebagai index
        data_df['Tanggal'] = pd.to_datetime(data_df['Tanggal'], format='%d-%b-%y')
        data_df.set_index('Tanggal', inplace=True)

        # Dropdown untuk memilih provinsi
        selected_province = st.selectbox("Pilih Provinsi", options=data_df.columns.tolist())
        
        if selected_province:
            # Menampilkan statistik deskriptif
            st.subheader(f"Statistika Deskriptif untuk {selected_province}")
            st.write(data_df[selected_province].describe())

            # Menampilkan plot line chart
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

# Halaman Pemetaan
def pemetaan():
    st.subheader("Pemetaan Clustering dengan DTW")
    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"], key="pemetaan_upload")

    if uploaded_file is not None:
        data_df = pd.read_csv(uploaded_file)
        data_df['Tanggal'] = pd.to_datetime(data_df['Tanggal'], format='%d-%b-%y')
        data_df.set_index('Tanggal', inplace=True)

        # Menghitung rata-rata per minggu dan bulan
        data_weekly = data_df.resample('W').mean()
        data_monthly = data_df.resample('M').mean()

        # Standarisasi data mingguan dan bulanan
        scaler = StandardScaler()
        data_weekly_standardized = pd.DataFrame(scaler.fit_transform(data_weekly), index=data_weekly.index, columns=data_weekly.columns)
        data_monthly_standardized = pd.DataFrame(scaler.fit_transform(data_monthly), index=data_monthly.index, columns=data_monthly.columns)

        # Hitung matriks biaya lokal dan akumulatif untuk data mingguan dan bulanan
        local_cost_matrix_weekly = compute_local_cost_matrix(data_weekly_standardized)
        accumulated_cost_matrix_weekly = compute_accumulated_cost_matrix(local_cost_matrix_weekly)

        local_cost_matrix_monthly = compute_local_cost_matrix(data_monthly_standardized)
        accumulated_cost_matrix_monthly = compute_accumulated_cost_matrix(local_cost_matrix_monthly)

        # Hitung matriks jarak DTW untuk data bulanan
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
                min_prev_cost = np.min([accumulated_cost_matrix[t-1, i, j], accumulated_cost_matrix[t-1, i, :].min(), accumulated_cost_matrix[t-1, :, j].min()])
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

# Fungsi utama aplikasi
def main():
    st.title("Aplikasi Statistika Deskriptif dan Pemetaan")
    page = st.sidebar.radio("Pilih Halaman", ("Statistika Deskriptif", "Pemetaan"))

    if page == "Statistika Deskriptif":
        statistik_deskriptif()
    elif page == "Pemetaan":
        pemetaan()

if __name__ == "__main__":
    main()
