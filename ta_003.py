import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn_extra.cluster import KMedoids
import geopandas as gpd
import json

# Function to upload CSV file
def upload_csv_file():
    uploaded_file = st.file_uploader("Unggah File CSV", type=['csv'])
    if uploaded_file is not None:
        data_df = pd.read_csv(uploaded_file)
        return data_df
    return None

# Function to upload GeoJSON file from GitHub
def upload_geojson_file():
    url = 'https://raw.githubusercontent.com/yourusername/yourrepository/main/indonesia-prov.geojson'  # Replace with your GitHub link
    try:
        with open(url) as f:
            data = json.load(f)
        gdf = gpd.GeoDataFrame.from_features(data['features'])
        return gdf
    except Exception as e:
        st.error(f"Error loading GeoJSON: {e}")
        return None

# Function for Statistika Deskriptif page
def statistika_deskriptif(data_df):
    st.subheader("Statistika Deskriptif")
    
    if data_df is not None:
        st.write(data_df.describe())

        # Line chart visualization
        st.subheader("Visualisasi Rata-rata Harga per Tanggal")
        selected_province = st.selectbox("Pilih Provinsi", data_df.columns)
        
        # Plot line chart for selected province
        fig, ax = plt.subplots()
        data_df[selected_province].plot(ax=ax)
        ax.set_title(f"Rata-rata Harga untuk {selected_province}")
        ax.set_xlabel("Tanggal")
        ax.set_ylabel("Harga")
        st.pyplot(fig)

# Function for Pemetaan page
def pemetaan(data_df):
    st.subheader("Pemetaan Data")

    if data_df is not None:
        # Handle missing data by forward filling
        data_df.fillna(method='ffill', inplace=True)

        # Convert date column to datetime if necessary
        data_df['Tanggal'] = pd.to_datetime(data_df['Tanggal'])
        data_daily = data_df.set_index('Tanggal').resample('D').mean()  # Resampling to daily data

        # Load GeoJSON file from GitHub
        gdf = upload_geojson_file()

        if gdf is not None:
            gdf = gdf.rename(columns={'Propinsi': 'Province'})  # Change according to the correct column name
            gdf['Province'] = gdf['Province'].str.upper().str.replace('.', '', regex=False).str.strip()

            # Create a map visualization
            st.subheader("Peta Provinsi")
            selected_date = st.date_input("Pilih Tanggal", value=data_daily.index[-1])
            daily_data = data_daily.loc[selected_date]

            # Merge data with GeoDataFrame
            gdf = gdf.merge(daily_data, left_on='Province', right_index=True, how='left')

            # Set color based on price
            gdf['color'] = gdf['Harga'].apply(lambda x: 'red' if x > daily_data.mean() else 'lightpink')
            
            # Plot map
            fig, ax = plt.subplots(1, 1, figsize=(12, 10))
            gdf.boundary.plot(ax=ax, linewidth=1, color='black')  # Plot boundaries
            gdf.plot(ax=ax, color=gdf['color'], edgecolor='black', alpha=0.7)  # Plot prices
            plt.title(f"Pemetaan Provinsi untuk Tanggal {selected_date}")
            st.pyplot(fig)

# Function for Pemetaan KMedoids page
def pemetaan_kmedoids(data_df):
    st.subheader("Pemetaan KMedoids")

    if data_df is not None:
        # Handle missing data by forward filling
        data_daily = data_df.fillna(method='ffill')

        # Standardization of data
        scaler = StandardScaler()
        data_daily_values = scaler.fit_transform(data_daily)

        # Number of clusters input
        n_clusters = st.slider("Pilih Jumlah Kluster", min_value=2, max_value=10, value=3)

        # Apply KMedoids clustering
        kmedoids = KMedoids(n_clusters=n_clusters, random_state=0, metric='euclidean')
        cluster_labels = kmedoids.fit_predict(data_daily_values)

        # Plot average prices for each cluster
        st.subheader("Visualisasi Rata-rata Harga untuk Setiap Kluster")

        # Create a DataFrame for plotting
        cluster_df = pd.DataFrame(data_daily_values, columns=data_daily.columns)
        cluster_df['Cluster'] = cluster_labels

        # Calculate average prices per cluster
        avg_prices_per_cluster = cluster_df.groupby('Cluster').mean().T

        # Plot average prices for each cluster
        fig, ax = plt.subplots(figsize=(10, 6))
        avg_prices_per_cluster.plot(ax=ax)
        ax.set_title("Rata-rata Harga per Kluster")
        ax.set_xlabel("Tanggal")
        ax.set_ylabel("Harga")
        ax.legend(title="Kluster", bbox_to_anchor=(1.05, 1), loc='upper left')

        st.pyplot(fig)

        # Load GeoJSON file from GitHub
        gdf = upload_geojson_file()

        if gdf is not None:
            gdf = gdf.rename(columns={'Propinsi': 'Province'})  # Change according to the correct column name
            gdf['Province'] = gdf['Province'].str.upper().str.replace('.', '', regex=False).str.strip()

            # Calculate cluster from clustering results
            cluster_labels = pd.Series(cluster_labels, index=data_daily.columns)
            cluster_labels = cluster_labels.reset_index()
            cluster_labels.columns = ['Province', 'Cluster']

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
            gdf = gdf.merge(cluster_labels, on='Province', how='left')

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

# Main function to control navigation between pages
def main():
    st.title("Aplikasi Pemetaan Clustering")

    with st.sidebar:
        selected_page = st.selectbox("Menu", ["Statistika Deskriptif", "Pemetaan", "Pemetaan KMedoids"])

    data_df = upload_csv_file()  # Upload CSV file

    if selected_page == "Statistika Deskriptif":
        statistika_deskriptif(data_df)
    elif selected_page == "Pemetaan":
        pemetaan(data_df)
    elif selected_page == "Pemetaan KMedoids":
        pemetaan_kmedoids(data_df)

if __name__ == "__main__":
    main()
