import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn_extra.cluster import KMedoids
from sklearn.preprocessing import StandardScaler
import geopandas as gpd
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
        province_options = [col for col in data_df.columns if col != 'Tanggal']
        selected_province = st.selectbox("Pilih Provinsi untuk Visualisasi", province_options)

        if selected_province:
            st.write(f"Rata-rata harga untuk provinsi: {selected_province}")
            data_df['Tanggal'] = pd.to_datetime(data_df['Tanggal'], format='%d-%b-%y', errors='coerce')
            data_df.set_index('Tanggal', inplace=True)

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(data_df.index, data_df[selected_province], label=selected_province)
            ax.set_title(f"Rata-rata Harga Harian - Provinsi {selected_province}")
            ax.set_xlabel("Tanggal")
            ax.set_ylabel("Harga")
            ax.legend()

            st.pyplot(fig)

# Pemetaan Page
def pemetaan(data_df):
    st.subheader("Pemetaan Data")

    if data_df is not None:
        data_df['Tanggal'] = pd.to_datetime(data_df['Tanggal'], format='%d-%b-%y', errors='coerce')
        data_df.set_index('Tanggal', inplace=True)

        # Handle missing data by forward filling
        data_df.fillna(method='ffill', inplace=True)

        # Dropdown for selecting a date
        selected_date = st.date_input("Pilih Tanggal", value=data_df.index.min().date())

        # Filter data by selected date
        if selected_date:
            selected_data = data_df.loc[selected_date.strftime('%Y-%m-%d')]
            st.write(f"Data untuk tanggal {selected_date}:")
            st.write(selected_data)

            # Visualization (You can modify this to plot on a map or other visual elements)
            st.line_chart(selected_data)

# Pemetaan KMedoids Page
def pemetaan_kmedoids(data_df):
    st.subheader("Pemetaan Clustering dengan KMedoids")

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

        # Dropdown for number of clusters
        n_clusters = st.slider("Pilih Jumlah Kluster", min_value=2, max_value=10, value=3)

        # Apply KMedoids clustering
        kmedoids = KMedoids(n_clusters=n_clusters, random_state=0, metric='euclidean')
        kmedoids.fit(data_daily_values.T)  # Transpose to cluster on provinces

        # Get cluster labels
        cluster_labels = kmedoids.labels_

        # Display Silhouette score
        silhouette_avg = silhouette_score(data_daily_values.T, cluster_labels)
        st.write(f"Silhouette Score untuk {n_clusters} kluster: {silhouette_avg:.2f}")

        # Create a DataFrame to show the clusters
        clustered_data = pd.DataFrame({
            'Province': data_daily.columns,
            'Cluster': cluster_labels
        })

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
            gdf.boundary.plot(ax=ax, linewidth=1, color='black')  # Plot boundaries
            gdf.plot(ax=ax, color=gdf['color'], edgecolor='black', alpha=0.7)  # Plot clusters
            plt.title("Pemetaan Provinsi Berdasarkan Kluster (KMedoids)")
            st.pyplot(fig)

# Main function to run the Streamlit app
def main():
    st.title("Aplikasi Pemodelan dan Pemetaan Data")
    
    # Sidebar menu for navigation
    with st.sidebar:
        selected_option = option_menu("Menu", ["Statistika Deskriptif", "Pemetaan", "Pemetaan KMedoids"], 
                                       icons=["bar-chart", "map", "cluster"], 
                                       menu_icon="cast", default_index=0)

    # Upload data file once
    data_df = upload_csv_file()

    # Call the appropriate page based on the selected option
    if selected_option == "Statistika Deskriptif":
        statistika_deskriptif(data_df)
    elif selected_option == "Pemetaan":
        pemetaan(data_df)
    elif selected_option == "Pemetaan KMedoids":
        pemetaan_kmedoids(data_df)

if __name__ == "__main__":
    main()
