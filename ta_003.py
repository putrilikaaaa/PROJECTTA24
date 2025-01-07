import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
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

# Function to compute DTW distance matrix for linkage
def compute_dtw_distance_matrix(data):
    num_series = data.shape[1]
    dtw_distance_matrix = np.zeros((num_series, num_series))

    for i in range(num_series):
        for j in range(i, num_series):
            # Using fastdtw to compute DTW distance
            distance, _ = fastdtw(data.iloc[:, i].values, data.iloc[:, j].values)
            dtw_distance_matrix[i, j] = distance
            dtw_distance_matrix[j, i] = distance  # Symmetry enforcement

    return dtw_distance_matrix

# Function to symmetrize a matrix (making it symmetric)
def symmetrize(matrix):
    return (matrix + matrix.T) / 2

# Statistika Deskriptif Page
def statistika_deskriptif(data_df):
    st.subheader("Halaman Statistika Deskriptif")
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

# Pemetaan Linkage Page
def pemetaan(data_df):
    st.subheader("Halaman Pemetaan dengan Metode Linkage")

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
        st.write(f"Jumlah cluster optimal berdasarkan Silhouette Score adalah: {optimal_n_clusters}")

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

        # GeoJSON visualization with cluster dropdown
        gdf = upload_geojson_file()
        if gdf is not None:
            gdf = gdf.rename(columns={'Propinsi': 'Province'})
            gdf['Province'] = gdf['Province'].str.upper().str.replace('.', '', regex=False).str.strip()

            # Ensure the Province column exists
            if 'Province' in clustered_data.columns:
                clustered_data['Province'] = clustered_data['Province'].str.upper().str.replace('.', '', regex=False).str.strip()
            else:
                st.error("Province column not found in clustered_data.")

            gdf['Province'] = gdf['Province'].replace({
                'DI ACEH': 'ACEH',
                'KEPULAUAN BANGKA BELITUNG': 'BANGKA BELITUNG',
                'NUSATENGGARA BARAT': 'N USA TENGGARA BARAT',
                'D.I YOGYAKARTA': 'DI YOGYAKARTA',
                'DAERAH ISTIMEWA YOGYAKARTA': 'DI YOGYAKARTA',
            })

            gdf = gdf[gdf['Province'].notna()]
            gdf = gdf.merge(clustered_data, on='Province', how='left')

            cluster_options = list(range(1, optimal_n_clusters + 1))
            selected_cluster = st.selectbox("Pilih cluster untuk Pemetaan", options=cluster_options)

            # Calculate average values for provinces in the selected cluster
            provinces_in_cluster = clustered_data[clustered_data['Cluster'] == selected_cluster]['Province']
            provinces_in_cluster = provinces_in_cluster.str.upper().str.replace('.', '', regex=False).str.strip()

            # Ensure the columns in data_to_plot are also transformed
            data_to_plot_original = pd.DataFrame(
                data_daily, 
                columns=data_daily.columns.str.upper().str.replace('.', '', regex=False).str.strip(),
                index=data_daily.index
            )
            data_to_plot_selected_cluster = data_to_plot_original[provinces_in_cluster].copy()

            # Calculate the average for each province in the selected cluster
            average_values = data_to_plot_selected_cluster.mean(axis=0)

            
            gdf['Average'] = gdf['Province'].map(average_values)

            # Create a heatmap based on the average values
            fig, ax = plt.subplots(1, 1, figsize=(12, 10))
            gdf.boundary.plot(ax=ax, linewidth=1, color='black')  # Garis batas
            gdf[gdf['Average'].notna()].plot(
                column='Average', 
                ax=ax, 
                legend=True,
                legend_kwds={
                    'label': "Rata-rata Nilai",
                    'orientation': "horizontal"
                },
                cmap='YlOrRd',  # Skema warna
                missing_kwds={"color": "lightgrey"}  # Warna jika data hilang
            )
            plt.title(f"Peta Provinsi per Cluster {selected_cluster} - Agglomerative (DTW)")
            st.pyplot(fig)

            # Calculate the average line across the selected cluster provinces
            average_line = data_to_plot_selected_cluster.mean(axis=1)

            st.write("Rentang nilai rata-rata:", average_values.min(), "-", average_values.max())

            # Plot the line chart for the selected cluster
            plt.figure(figsize=(12, 6))
            for province in provinces_in_cluster:
                plt.plot(data_to_plot_selected_cluster.index, data_to_plot_selected_cluster[province], color='gray', alpha=0.5)
            plt.plot(average_line.index, average_line, color='red', linewidth=2, label='Rata-rata Provinsi dalam Cluster')
            plt.title(f'Line Chart untuk Cluster {selected_cluster}')
            plt.xlabel('Tanggal')
            plt.ylabel('Nilai')
            plt.legend()
            st.pyplot(plt)
            

            
# Function to compute DTW distance matrix using fastdtw for medoids
def compute_dtw_distance_matrix(data):
    num_series = data.shape[1]
    dtw_distance_matrix = np.zeros((num_series, num_series))

    for i in range(num_series):
        for j in range(i, num_series):
            # Using fastdtw to compute DTW distance
            distance, _ = fastdtw(data[:, i], data[:, j])  # Access numpy array columns
            dtw_distance_matrix[i, j] = distance
            dtw_distance_matrix[j, i] = distance  # DTW distance is symmetric

    return dtw_distance_matrix

# Pemetaan KMedoids Page
def pemetaan_kmedoids(data_df):
    st.subheader("Halaman Pemetaan dengan Metode K-Medoids")

    if data_df is not None:
        data_df['Tanggal'] = pd.to_datetime(data_df['Tanggal'], format='%d-%b-%y', errors='coerce')
        data_df.set_index('Tanggal', inplace=True)

        data_daily = data_df.resample('D').mean()
        data_daily.fillna(method='ffill', inplace=True)

        # Scaling the data using MinMaxScaler
        scaler = MinMaxScaler()
        data_daily_values = scaler.fit_transform(data_daily)

        # Compute DTW distance matrix using fastdtw
        dtw_distance_matrix_daily = compute_dtw_distance_matrix(data_daily_values)

        # K-Medoids clustering using DTW distance matrix
        max_n_clusters = 10
        silhouette_scores = {}
        cluster_labels_dict = {}

        for n_clusters in range(2, max_n_clusters + 1):
            kmedoids = KMedoids(n_clusters=n_clusters, metric='precomputed', init='k-medoids++', random_state=42)
            kmedoids.fit(dtw_distance_matrix_daily)
            labels = kmedoids.labels_
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

        # Get cluster labels and map them starting from 1
        cluster_labels = cluster_labels_dict[optimal_n_clusters] + 1
        clustered_data = pd.DataFrame({
            'Province': data_daily.columns,
            'Cluster': cluster_labels
        })

        st.subheader("Tabel Label Cluster Setiap Provinsi")
        st.write(clustered_data)

        # GeoJSON visualization with cluster dropdown
        gdf = upload_geojson_file()
        if gdf is not None:
            gdf = gdf.rename(columns={'Propinsi': 'Province'})
            gdf['Province'] = gdf['Province'].str.upper().str.replace('.', '', regex=False).str.strip()

            # Ensure the Province column exists
            if 'Province' in clustered_data.columns:
                clustered_data['Province'] = clustered_data['Province'].str.upper().str.replace('.', '', regex=False).str.strip()
            else:
                st.error("Kolom 'Province' tidak ditemukan dalam clustered_data.")

            gdf['Province'] = gdf['Province'].replace({
                'DI ACEH': 'ACEH',
                'KEPULAUAN BANGKA BELITUNG': 'BANGKA BELITUNG',
                'NUSATENGGARA BARAT': 'NUSA TENGGARA BARAT',
                'D.I YOGYAKARTA': 'DI YOGYAKARTA',
                'DAERAH ISTIMEWA YOGYAKARTA': 'DI YOGYAKARTA',
            })

            gdf = gdf[gdf['Province'].notna()]
            gdf = gdf.merge(clustered_data, on='Province', how='left')

            cluster_options = list(range(1, optimal_n_clusters + 1))
            selected_cluster = st.selectbox("Pilih Kluster untuk Pemetaan", options=cluster_options)

         # Calculate average values for provinces in the selected cluster
            provinces_in_cluster = clustered_data[clustered_data['Cluster'] == selected_cluster]['Province']
            provinces_in_cluster = provinces_in_cluster.str.upper().str.replace('.', '', regex=False).str.strip()

            # Ensure the columns in data_to_plot are also transformed
            data_to_plot_original = pd.DataFrame(
                data_daily, 
                columns=data_daily.columns.str.upper().str.replace('.', '', regex=False).str.strip(),
                index=data_daily.index
            )
            data_to_plot_selected_cluster = data_to_plot_original[provinces_in_cluster].copy()

            # Calculate the average for each province in the selected cluster
            average_values = data_to_plot_selected_cluster.mean(axis=0)

            
            gdf['Average'] = gdf['Province'].map(average_values)

            # Create a heatmap based on the average values
            fig, ax = plt.subplots(1, 1, figsize=(12, 10))
            gdf.boundary.plot(ax=ax, linewidth=1, color='black')  # Garis batas
            gdf[gdf['Average'].notna()].plot(
                column='Average', 
                ax=ax, 
                legend=True,
                legend_kwds={
                    'label': "Rata-rata Nilai",
                    'orientation': "horizontal"
                },
                cmap='YlOrRd',  # Skema warna
                missing_kwds={"color": "lightgrey"}  # Warna jika data hilang
            )
            plt.title(f"Peta Provinsi per Cluster {selected_cluster} - Agglomerative (DTW)")
            st.pyplot(fig)

            # Calculate the average line across the selected cluster provinces
            average_line = data_to_plot_selected_cluster.mean(axis=1)

            st.write("Rentang nilai rata-rata:", average_values.min(), "-", average_values.max())

            # Plot the line chart for the selected cluster
            plt.figure(figsize=(12, 6))
            for province in provinces_in_cluster:
                plt.plot(data_to_plot_selected_cluster.index, data_to_plot_selected_cluster[province], color='gray', alpha=0.5)
            plt.plot(average_line.index, average_line, color='red', linewidth=2, label='Rata-rata Provinsi dalam Cluster')
            plt.title(f'Line Chart untuk Cluster {selected_cluster}')
            plt.xlabel('Tanggal')
            plt.ylabel('Nilai')
            plt.legend()
            st.pyplot(plt)
# Sidebar options
selected = option_menu(
    menu_title=None,
    options=["Home Page", "Statistika Deskriptif", "Pemetaan Linkage", "Pemetaan KMedoids"],
    icons=["house", "bar-chart", "map", "map"],
    default_index=0,
    orientation="horizontal",
    styles={
        "nav-link": {"--hover-color": "#eee"},
        "icon": {"color": "white"},
        "nav-link-selected": {"background-color": "#2C6DD5"},
        "nav-link": {"font-size": "18px", "text-align": "center", "margin": "0px", "--hover-color": "#2C6DD5"},
        "nav-link-selected": {"background-color": "#2C6DD5"},
    }
)

# Load sample data for pages
def load_data():
    url = 'https://raw.githubusercontent.com/putrilikaaaa/PROJECTTA24/main/DATAFILE.csv'
    data = pd.read_csv(url)
    return data

if selected == "Home Page":
    st.title("Selamat Datang di Aplikasi Pengelompokkan dan Pemetaan Provinsi Indonesia")
    st.markdown("""
    <div style="text-align: justify;">
    Aplikasi ini dirancang untuk mengetahui pengelompokkan daerah provinsi di Indonesia berdasarkan pola waktunya.
    Metode pengelompokkan yang digunakan pada aplikasi ini adalah menggunakan jarak <em><strong>Dynamic Time Warping (DTW)</strong></em> dan
    metode pengelompokkan secara hierarki dengan menggunakan <em><strong>Single Linkage</strong></em>, <em><strong>Complete Linkage</strong></em>, dan
    <em><strong>Average Linkage</strong></em>, serta pengelompokkan secara non-hierarki dengan menggunakan <em><strong>K-Medoids</strong></em>.
    </div>
    """, unsafe_allow_html=True)

    # Panduan Pengguna section with download button
    st.subheader("Panduan Pengguna")
    st.markdown("""
    <div style="text-align: justify;">
    1. Download Template CSV dengan klik tombol berikut. Sesuaikan periode waktunya dengan periode waktu data anda dan jangan merubah nama provinsi. Data yang dimasukkan merupakan data runtun waktu seperti data nilai produksi, harga komoditas, temperatur udara, curah hujan, dan lainnya selama beberapa periode waktu.
    </div>
    """, unsafe_allow_html=True)

    # Download the template file
    template_url = 'https://github.com/putrilikaaaa/PROJECTTA24/raw/main/TEMPLATE.csv'
    response = requests.get(template_url)
    template_data = response.content

    # Add the download button
    st.download_button(
    label="Download Template CSV",
    data=template_data,
    file_name="TEMPLATE.csv",
    mime="text/csv"
    )

    # Add the second point and image
    st.markdown("""
    <div style="text-align: justify;">
    2. Klik halaman Statistika Deskriptif pada menu bagian atas halaman untuk melihat kenaikan dan penurunan nilai pada setiap periode waktu. Upload file pada bagian "Upload file CSV".
    </div>
    """, unsafe_allow_html=True)

    # Display the first image
    st.markdown("""
    <div style="border: 2px solid black; display: inline-block; padding: 5px;">
    <img src="https://github.com/putrilikaaaa/PROJECTTA24/raw/main/Page%20SD%201.png" alt="Statistika Deskriptif" width="600">
    </div>
    """, unsafe_allow_html=True)

    # Add additional explanation below the first image
    st.markdown("""
    <div style="text-align: justify;">
    Setelah mengupload data, akan muncul tampilan seperti di bawah ini. Anda dapat memilih provinsi pada dropdown dan visualisasi kenaikan dan penurunan nilai pada setiap provinsi akan disajikan oleh aplikasi.
    </div>
    """, unsafe_allow_html=True)

    # Display the second image with the raw URL
    st.markdown("""
    <div style="border: 2px solid black; display: inline-block; padding: 5px;">
    <img src="https://raw.githubusercontent.com/putrilikaaaa/PROJECTTA24/main/Page%20SD%202.png" alt="Statistika Deskriptif" width="600">
    </div>
    """, unsafe_allow_html=True)

    # Add additional explanation for the "Pemetaan Linkage" section
    st.markdown("""
    <div style="text-align: justify;">
    3. Klik halaman Pemetaan Linkage pada menu bagian atas halaman untuk melihat pemetaan dan pengelompokkan provinsi berdasarkan nilai dan periode waktu. Upload ulang file seperti petunjuk pada halaman sebelumnya. Lalu akan muncul tampilan seperti di bawah ini.
    </div>
    """, unsafe_allow_html=True)

    # Display the additional image for "Pemetaan Linkage"
    st.markdown("""
    <div style="border: 2px solid black; display: inline-block; padding: 5px;">
    <img src="https://raw.githubusercontent.com/putrilikaaaa/PROJECTTA24/main/Page%20PL%206.png" alt="Statistika Deskriptif" width="600">
    </div>
    """, unsafe_allow_html=True)

    # Display the image for the "Pemetaan Linkage" section
    st.markdown("""
    <div style="border: 2px solid black; display: inline-block; padding: 5px;">
    <img src="https://raw.githubusercontent.com/putrilikaaaa/PROJECTTA24/main/Page%20PL%202.png" alt="Statistika Deskriptif" width="600">
    </div>
    """, unsafe_allow_html=True)

    # Add the additional text below the image
    st.markdown("""
    <div style="text-align: justify;">
    Silahkan pilih metode yang ingin digunakan pada dropdown. Lalu akan muncul Silhouette Score Plot, Dendogram, dan Peta Indonedia seperti tampilan di bawah ini.
    </div>
    """, unsafe_allow_html=True)

    # Display the first image
    st.markdown("""
    <div style="border: 2px solid black; display: inline-block; padding: 5px;">
    <img src="https://raw.githubusercontent.com/putrilikaaaa/PROJECTTA24/main/Page%20PL%203.png" alt="Statistika Deskriptif" width="600">
    </div>
    """, unsafe_allow_html=True)

    # Add text below the first image
    st.markdown("""
    <div style="text-align: justify;">
    Silhouette Score tertinggi menunjukkan jumlah klaster yang optimum. Pada contoh di atas menunjukkan bahwa klaster optimum berjumlah 2 klaster karena memiliki Silhouette Score tertinggi, yaitu sebesar 0,46.
    </div>
    """, unsafe_allow_html=True)

    # Display the second image
    st.markdown("""
    <div style="border: 2px solid black; display: inline-block; padding: 5px;">
    <img src="https://raw.githubusercontent.com/putrilikaaaa/PROJECTTA24/main/Page%20PL%204.png" alt="Statistika Deskriptif" width="600">
    </div>
    """, unsafe_allow_html=True)

    # Add text below the second image
    st.markdown("""
    <div style="text-align: justify;">
    Dendogram menunjukkan anggota klaster berdasarkan tingkatan atau hierarkinya. Pada gambar di atas anggota klaster pertama memiliki garis berwarna jingga dan anggota klaster kedua memiliki garis berwarna hijau.
    </div>
    """, unsafe_allow_html=True)

    # Display the third image
    st.markdown("""
    <div style="border: 2px solid black; display: inline-block; padding: 5px;">
    <img src="https://raw.githubusercontent.com/putrilikaaaa/PROJECTTA24/main/PL%207.png" alt="Statistika Deskriptif" width="600">
    </div>
    """, unsafe_allow_html=True)

    # Add text below the third image
    st.markdown("""
    <div style="text-align: justify;">
    Terdapat dropdown yang mana pengguna dapat memilih kelompok mana yang mau ditampilkan. Semakin gelap warna suatu provinsi, maka semakin tinggi rata-rata harganya.
    <br><br>
    </div>
     """, unsafe_allow_html=True)

    # Display the third image
    st.markdown("""
    <div style="border: 2px solid black; display: inline-block; padding: 5px;">
    <img src="https://raw.githubusercontent.com/putrilikaaaa/PROJECTTA24/main/PL%208.png" alt="Statistika Deskriptif" width="600">
    </div>
    """, unsafe_allow_html=True)

    # Add text below the third image
    st.markdown("""
    <div style="text-align: justify;">
    Terdapat linechart yang mana berisi pola harga dari setiap provinsi yang berada pada kelompok yang dipilih yang ditunjukkan oleh garis berwarna abu-abu dan rata-rata pola harganya ditunjukkan oleh garis berwarna merah.
    <br><br>
    </div>
     """, unsafe_allow_html=True) 

    # Add text for "Pemetaan KMedoids"
    st.markdown("""
    <div style="text-align: justify;">
    3. Klik halaman Pemetaan KMedoids pada menu bagian atas halaman untuk melihat pemetaan dan pengelompokkan provinsi berdasarkan nilai dan periode waktu. Petunjuknya sama dengan halaman Pemetaan Linkage hanya saja tidak terdapat dendogram pada halaman ini karena metode KMedoids merupakan metode non-hierarki atau tidak memiliki tingkatan.
    </div>

   """, unsafe_allow_html=True)

    # Display the image for "Pemetaan KMedoids"
    st.markdown("""
    <div style="border: 2px solid black; display: inline-block; padding: 5px;">
    <img src="https://raw.githubusercontent.com/putrilikaaaa/PROJECTTA24/main/Page%20PK%201.png" alt="Statistika Deskriptif" width="600">
    </div>
    """, unsafe_allow_html=True)


elif selected == "Statistika Deskriptif":
    data_df = upload_csv_file()  # File upload for Statistika Deskriptif
    statistika_deskriptif(data_df)

elif selected == "Pemetaan Linkage":
    data_df = upload_csv_file()  # File upload for Pemetaan Linkage
    pemetaan(data_df)

elif selected == "Pemetaan KMedoids":
    data_df = upload_csv_file()  # File upload for Pemetaan KMedoids
    pemetaan_kmedoids(data_df)
