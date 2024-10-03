import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from sklearn.preprocessing import StandardScaler
from streamlit_option_menu import option_menu
import plotly.graph_objects as go

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

            # Plot average prices for the selected province using Plotly
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data_df.index, y=data_df[selected_province], mode='lines+markers', name=selected_province))
            fig.update_layout(title=f"Rata-rata Harga Harian - Provinsi {selected_province}",
                              xaxis_title="Tanggal",
                              yaxis_title="Harga",
                              hovermode="x unified")
            st.plotly_chart(fig)

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
            score = silhouette_score(dtw_distance_matrix_daily, labels, metric='precomputed')
            silhouette_scores[n_clusters] = score
            cluster_labels_dict[n_clusters] = labels

        # Plot Silhouette Scores using Plotly
        fig_silhouette = go.Figure()
        fig_silhouette.add_trace(go.Scatter(x=list(silhouette_scores.keys()), y=list(silhouette_scores.values()), mode='lines+markers'))
        fig_silhouette.update_layout(title='Silhouette Score vs. Number of Clusters (Data Harian)',
                                      xaxis_title='Number of Clusters',
                                      yaxis_title='Silhouette Score',
                                      xaxis=dict(tickmode='linear', tick0=2, dtick=1),
                                      yaxis=dict(range=[0, 1]))
        st.plotly_chart(fig_silhouette)

        # Determine optimal number of clusters
        optimal_n_clusters = max(silhouette_scores, key=silhouette_scores.get)
        st.write(f"Jumlah kluster optimal berdasarkan Silhouette Score adalah: {optimal_n_clusters}")

        # Clustering and dendrogram
        condensed_dtw_distance_matrix = squareform(dtw_distance_matrix_daily)
        Z = linkage(condensed_dtw_distance_matrix, method=linkage_method)

        # Plot dendrogram using Plotly
        fig_dendrogram = go.Figure()
        dendro_data = dendrogram(Z, no_plot=True)  # Get dendrogram data without plotting
        fig_dendrogram.add_trace(go.Scatter(x=dendro_data['icoord'], y=dendro_data['dcoord'], mode='lines'))
        fig_dendrogram.update_layout(title=f'Dendrogram Clustering dengan DTW (Data Harian) - Linkage: {linkage_method.capitalize()}',
                                      xaxis_title='Provinsi',
                                      yaxis_title='Jarak DTW')
        st.plotly_chart(fig_dendrogram)

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

            # Plot map using Plotly
            fig_map = go.Figure()
            for i, row in gdf.iterrows():
                fig_map.add_trace(go.Choropleth(
                    locations=[row['Province']],
                    locationmode='country names',
                    z=[row['Cluster']],
                    hoverinfo='location+z',
                    colorscale=[[0, 'red'], [0.5, 'yellow'], [1, 'green']],
                    colorbar=dict(title='Cluster'),
                    showscale=False,
                    marker=dict(line=dict(width=0)),
                ))

            fig_map.update_layout(title='Pemetaan Kluster Provinsi', geo=dict(showland=True))
            st.plotly_chart(fig_map)

# Function to compute local cost matrix for DTW
def compute_local_cost_matrix(data_df: pd.DataFrame) -> np.array:
    num_provinces = data_df.shape[1]
    local_cost_matrix = np.zeros((num_provinces, num_provinces))

    for i in range(num_provinces):
        for j in range(num_provinces):
            if i != j:
                cost = np.sum(np.abs(data_df.iloc[:, i] - data_df.iloc[:, j]))
                local_cost_matrix[i, j] = cost

    return local_cost_matrix

# Function to compute accumulated cost matrix for DTW
def compute_accumulated_cost_matrix(local_cost_matrix: np.array) -> np.array:
    num_provinces = local_cost_matrix.shape[0]
    accumulated_cost_matrix = np.zeros_like(local_cost_matrix)

    # Initialize first row and column
    accumulated_cost_matrix[0, 0] = local_cost_matrix[0, 0]

    for j in range(1, num_provinces):
        accumulated_cost_matrix[0, j] = accumulated_cost_matrix[0, j - 1] + local_cost_matrix[0, j]
    
    for i in range(1, num_provinces):
        accumulated_cost_matrix[i, 0] = accumulated_cost_matrix[i - 1, 0] + local_cost_matrix[i, 0]

    for i in range(1, num_provinces):
        for j in range(1, num_provinces):
            accumulated_cost_matrix[i, j] = local_cost_matrix[i, j] + min(
                accumulated_cost_matrix[i - 1, j],     # from above
                accumulated_cost_matrix[i, j - 1],     # from left
                accumulated_cost_matrix[i - 1, j - 1]  # from diagonal
            )

    return accumulated_cost_matrix

# Function to compute DTW distance matrix
def compute_dtw_distance_matrix(accumulated_cost_matrix: np.array) -> np.array:
    num_provinces = accumulated_cost_matrix.shape[0]
    dtw_distance_matrix = np.zeros((num_provinces, num_provinces))

    for i in range(num_provinces):
        for j in range(num_provinces):
            dtw_distance_matrix[i, j] = accumulated_cost_matrix[i, j]

    return dtw_distance_matrix

# Main application
def main():
    st.title("Aplikasi Pemetaan dan Analisis Data")
    
    selected_page = option_menu("Menu", ["Statistika Deskriptif", "Pemetaan"], 
                                  icons=["clipboard-data", "geo-alt"], 
                                  menu_icon="cast", default_index=0)

    data_df = upload_csv_file()

    if selected_page == "Statistika Deskriptif":
        statistika_deskriptif(data_df)
    elif selected_page == "Pemetaan":
        pemetaan(data_df)

if __name__ == "__main__":
    main()
