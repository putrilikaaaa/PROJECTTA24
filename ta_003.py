import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn_extra.cluster import KMedoids
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
import geopandas as gpd

# Function to upload CSV file
def upload_csv_file():
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        return data
    return None

# Function to upload GeoJSON file from GitHub
def upload_geojson_file():
    geojson_url = "https://raw.githubusercontent.com/yourusername/yourrepo/main/indonesia-prov.geojson"
    gdf = gpd.read_file(geojson_url)
    return gdf

# Function for descriptive statistics
def statistika_deskriptif(data_df):
    st.subheader("Statistika Deskriptif")
    st.write(data_df.describe())

# Function for KMedoids clustering
def pemetaan_kmedoids(data_df):
    st.subheader("Pemetaan KMedoids")
    
    # Assuming data_df is organized with dates as rows and provinces as columns
    data_daily = data_df.set_index('Tanggal').T  # Transpose for KMedoids clustering

    # Resample data to daily frequency and fill missing values
    data_daily = data_daily.resample('D').mean()

    # Handle missing data by forward filling
    data_daily.fillna(method='ffill', inplace=True)

    # Standardization of data
    scaler = StandardScaler()
    data_daily_values = scaler.fit_transform(data_daily)

    # KMedoids Clustering
    n_clusters = st.slider("Pilih Jumlah Kluster KMedoids", 2, 10, 3)
    kmedoids = KMedoids(n_clusters=n_clusters, random_state=42, metric='euclidean')
    kmedoids.fit(data_daily_values)

    # Display cluster labels
    data_daily['Cluster'] = kmedoids.labels_

    # Load GeoJSON file from GitHub
    gdf = upload_geojson_file()

    if gdf is not None:
        gdf = gdf.rename(columns={'Propinsi': 'Province'})  # Change according to the correct column name
        gdf['Province'] = gdf['Province'].str.upper().str.replace('.', '', regex=False).str.strip()

        # Prepare clustered data DataFrame correctly
        clustered_data = pd.DataFrame({
            'Province': data_daily.index,  # Ensure we get the province names from the index
            'Cluster': data_daily['Cluster']
        })

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
        plt.title("Pemetaan Provinsi Berdasarkan Kluster KMedoids")
        st.pyplot(fig)

# Main function to control the app
def main():
    st.title("Aplikasi Clustering Data Provinsi")

    # Sidebar for navigation
    with st.sidebar:
        selected = option_menu("Menu", ["Statistika Deskriptif", "Pemetaan Linkage", "Pemetaan KMedoids"], 
                                icons=["graph-up-arrow", "map", "map"], menu_icon="cast", default_index=0)

    # Upload CSV file
    data_df = upload_csv_file()

    if data_df is not None:
        if selected == "Statistika Deskriptif":
            statistika_deskriptif(data_df)
        elif selected == "Pemetaan Linkage":
            # Include your pemetaan_linkage function here
            pass  # Placeholder for your linkage code
        elif selected == "Pemetaan KMedoids":
            pemetaan_kmedoids(data_df)

if __name__ == "__main__":
    main()
