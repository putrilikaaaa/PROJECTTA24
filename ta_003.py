import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn_extra.cluster import KMedoids  # Importing KMedoids
from sklearn.preprocessing import MinMaxScaler  # Importing MinMaxScaler for normalization
import geopandas as gpd

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

# Pemetaan KMedoids Page
def pemetaan_kmedoids(data_df):
    st.subheader("Pemetaan KMedoids")

    if data_df is not None:
        # Convert 'Tanggal' to datetime and set as index
        data_df['Tanggal'] = pd.to_datetime(data_df['Tanggal'], format='%d-%b-%y', errors='coerce')
        data_df.set_index('Tanggal', inplace=True)

        # Calculate daily averages and handle missing data by forward filling
        data_daily = data_df.resample('D').mean()
        data_daily.fillna(method='ffill', inplace=True)

        # Normalize the data
        scaler = MinMaxScaler()
        data_daily_values = scaler.fit_transform(data_daily)

        # Perform KMedoids clustering
        n_clusters = st.slider("Pilih jumlah kluster:", min_value=2, max_value=10, value=3)
        kmedoids = KMedoids(n_clusters=n_clusters, metric="euclidean", random_state=42)
        labels = kmedoids.fit_predict(data_daily_values.T)

        # Create a DataFrame to store the clustering results
        cluster_data = pd.DataFrame({'Province': data_daily.columns, 'Cluster': labels})

        # Display the cluster table
        st.write("Tabel provinsi per kluster:")
        st.write(cluster_data)

        # Plot clusters on a map
        gdf = upload_geojson_file()
        if gdf is not None:
            # Standardize province names in GeoDataFrame and clustered data
            gdf = gdf.rename(columns={'Propinsi': 'Province'})
            gdf['Province'] = gdf['Province'].str.upper().str.replace('.', '', regex=False).str.strip()

            cluster_data['Province'] = cluster_data['Province'].str.upper().str.replace('.', '', regex=False).str.strip()

            # Handle inconsistent names between the datasets (for example)
            gdf['Province'] = gdf['Province'].replace({
                'KEPULAUAN BANGKA BELITUNG': 'BANGKA BELITUNG',
                'NUSATENGGARA BARAT': 'NUSA TENGGARA BARAT',
                'D.I YOGYAKARTA': 'DI YOGYAKARTA',
                'DAERAH ISTIMEWA YOGYAKARTA': 'DI YOGYAKARTA',
            })

            # Merge cluster data with GeoDataFrame
            gdf = gdf.merge(cluster_data, on="Province", how="left")

            # Assign colors to clusters
            gdf['color'] = gdf['Cluster'].map({
                0: 'red', 1: 'yellow', 2: 'green', 3: 'blue', 4: 'purple',
                5: 'orange', 6: 'pink', 7: 'brown', 8: 'cyan', 9: 'magenta'
            }).fillna('grey')  # Provinces with no cluster get grey

            # Check for provinces not in clusters (colored grey)
            grey_provinces = gdf[gdf['color'] == 'grey']['Province'].tolist()
            if grey_provinces:
                st.subheader("Provinsi yang Tidak Termasuk dalam Kluster:")
                st.write(grey_provinces)
            else:
                st.write("Semua provinsi termasuk dalam kluster.")

            # Plot the map
            fig, ax = plt.subplots(1, 1, figsize=(12, 10))
            gdf.boundary.plot(ax=ax, linewidth=1, color='black')
            gdf.plot(ax=ax, color=gdf['color'], edgecolor='black', alpha=0.7)
            plt.title("Pemetaan KMedoids")
            st.pyplot(fig)

# Main function
def main():
    st.sidebar.title("Menu")
    choice = st.sidebar.selectbox("Pilih halaman", ["Pemetaan KMedoids"])

    # Upload CSV file
    data_df = upload_csv_file()

    # Display selected page
    if choice == "Pemetaan KMedoids":
        pemetaan_kmedoids(data_df)

if __name__ == "__main__":
    main()
