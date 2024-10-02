import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
import geopandas as gpd
import folium
from folium import Choropleth
from streamlit_option_menu import option_menu

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

# Ensure DTW distance matrix is symmetric
def symmetrize(matrix):
    return (matrix + matrix.T) / 2

# Standardization using the provided formulas
def standardize(data_df: pd.DataFrame) -> pd.DataFrame:
    n = len(data_df)
    Z_bar = data_df.mean()
    S_z = np.sqrt(((data_df - Z_bar) ** 2).mean())
    standardized_data = (data_df - Z_bar) / S_z
    return standardized_data

# Function to compute local cost matrix for DTW
def compute_local_cost_matrix(data_df: pd.DataFrame) -> np.array:
    num_time_points, num_provinces = data_df.shape
    local_cost_matrix = np.zeros((num_time_points, num_provinces, num_provinces))

    for i in range(num_provinces):
        for j in range(num_provinces):
            if i != j:
                for t in range(num_time_points):
                    local_cost_matrix[t, i, j] = np.abs(data_df.iloc[t, i] - data_df.iloc[t, j])

    return local_cost_matrix

# Function to compute accumulated cost matrix for DTW
def compute_accumulated_cost_matrix(local_cost_matrix: np.array) -> np.array:
    num_time_points, num_provinces, _ = local_cost_matrix.shape
    accumulated_cost_matrix = np.zeros((num_time_points, num_provinces, num_provinces))

    for i in range(num_provinces):
        for j in range(num_provinces):
            accumulated_cost_matrix[0, i, j] = local_cost_matrix[0, i, j]

    for t in range(1, num_time_points):
        for i in range(num_provinces):
            for j in range(num_provinces):
                if i == j:
                    # Check bounds for i-1 and i+1
                    min_prev_accumulated = accumulated_cost_matrix[t-1, i, j]
                    if i > 0:
                        min_prev_accumulated = min(min_prev_accumulated, accumulated_cost_matrix[t-1, i-1, j])
                    if i < num_provinces - 1:
                        min_prev_accumulated = min(min_prev_accumulated, accumulated_cost_matrix[t-1, i+1, j])
                    accumulated_cost_matrix[t, i, j] = local_cost_matrix[t, i, j] + min_prev_accumulated
                else:
                    # Check bounds for i-1 and i+1
                    min_prev_accumulated = min(accumulated_cost_matrix[t-1, i, j],
                                               accumulated_cost_matrix[t-1, i-1, j] if i > 0 else np.inf,
                                               accumulated_cost_matrix[t-1, i+1, j] if i < num_provinces - 1 else np.inf)
                    accumulated_cost_matrix[t, i, j] = local_cost_matrix[t, i, j] + min_prev_accumulated

    return accumulated_cost_matrix

# Function to compute DTW distance matrix
def compute_dtw_distance_matrix(accumulated_cost_matrix: np.array) -> np.array:
    num_provinces = accumulated_cost_matrix.shape[1]
    dtw_distance_matrix = np.zeros((num_provinces, num_provinces))

    for i in range(num_provinces):
        for j in range(num_provinces):
            if i != j:
                dtw_distance_matrix[i, j] = accumulated_cost_matrix[-1, i, j]

    return dtw_distance_matrix

# Statistika Deskriptif Page
def statistika_deskriptif(data_df):
    st.subheader("Statistika Deskriptif")

    if data_df is not None:
        st.write("Data yang diunggah:")
        st.write(data_df)

        # Statistika deskriptif
        st.write("Statistika deskriptif data:")
        st.write(data_df.describe())

        # Dropdown untuk memilih provinsi, kecuali kolom 'Tanggal'
        province_options = [col for col in data_df.columns if col != 'Tanggal']  # Menghilangkan 'Tanggal' dari pilihan
        selected_province = st.selectbox("Pilih Provinsi untuk Visualisasi", province_options)

        if selected_province:
            # Visualisasi data untuk provinsi terpilih
            st.write(f"Rata-rata harga untuk provinsi: {selected_province}")
            data_df['Tanggal'] = pd.to_datetime(data_df['Tanggal'], format='%d-%b-%y', errors='coerce')
            data_df.set_index('Tanggal', inplace=True)

            # Plot average prices for the selected province
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(data_df.index, data_df[selected_province], label=selected_province)
            ax.set_title(f"Rata-rata Harga Harian - Provinsi {selected_province}")
            ax.set_xlabel("Tanggal")
            ax.set_ylabel("Harga")
            ax.legend()

            st.pyplot(fig)

# Pemetaan Page
def pemetaan(data_df):
    st.subheader("Pemetaan Clustering dengan DTW")

    if data_df is not None:
        data_df['Tanggal'] = pd.to_datetime(data_df['Tanggal'], format='%d-%b-%y', errors='coerce')
        data_df.set_index('Tanggal', inplace=True)

        # Calculate daily averages
        data_daily = data_df.resample('D').mean()

        # Handle missing data by forward filling
        data_daily.fillna(method='ffill', inplace=True)

        # Standardize daily data
        data_daily_standardized = standardize(data_daily)

        # Dropdown for choosing linkage method
        linkage_method = st.selectbox("Pilih Metode Linkage", options=["complete", "single", "average"])

        # Compute local cost matrix and accumulated cost matrix
        local_cost_matrix_daily = compute_local_cost_matrix(data_daily_standardized)
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
        dendrogram(Z, labels=data_daily_standardized.columns, leaf_rotation=90)
        plt.title(f'Dendrogram Clustering dengan DTW (Data Harian) - Linkage: {linkage_method.capitalize()}')
        plt.xlabel('Provinsi')
        plt.ylabel('Jarak DTW')
        st.pyplot(plt)

        # Create a map visualization
        gdf = upload_geojson_file()
        cluster_labels = cluster_labels_dict[optimal_n_clusters]
        cluster_labels_df = pd.DataFrame({'Province': data_daily_standardized.columns, 'Cluster': cluster_labels})
        
        # Merge the GeoDataFrame with the clustering results
        gdf = gdf.merge(cluster_labels_df, left_on='properties.name', right_on='Province', how='left')

        # Create a Folium map
        m = folium.Map(location=[-5, 120], zoom_start=5)

        # Add Choropleth layer to the map
        Choropleth(
            geo_data=gdf,
            name='choropleth',
            data=cluster_labels_df,
            columns=['Province', 'Cluster'],
            key_on='feature.properties.name',
            fill_color='YlGn',
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name='Cluster'
        ).add_to(m)

        folium.LayerControl().add_to(m)
        st.subheader("Peta Kluster Provinsi")
        st.markdown(m._repr_html_(), unsafe_allow_html=True)

        # Show cluster results
        st.write("Daftar Provinsi per Kluster:")
        st.write(cluster_labels_df.groupby('Cluster')['Province'].apply(list))

# Main Streamlit app
def main():
    st.title("Analisis Data Provinsi di Indonesia")
    
    # Sidebar navigation
    with st.sidebar:
        selected_option = option_menu("Menu", ["Statistika Deskriptif", "Pemetaan"], 
                                       icons=["bar-chart", "map"], 
                                       menu_icon="cast", default_index=0)

    # Upload data
    data_df = upload_csv_file()

    # Handle page selection
    if selected_option == "Statistika Deskriptif":
        statistika_deskriptif(data_df)
    elif selected_option == "Pemetaan":
        pemetaan(data_df)

if __name__ == "__main__":
    main()
