import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load and preprocess the data
@st.cache
def load_data(filepath):
    data = pd.read_csv(filepath)
    # Remove 'Tanggal' if it exists or handle it appropriately
    if 'Tanggal' in data.columns:
        data = data.drop(columns=['Tanggal'])
    return data

# Function for scaling data
def scale_data(data, scaling_method='standard'):
    if scaling_method == 'standard':
        scaler = StandardScaler()
    elif scaling_method == 'minmax':
        scaler = MinMaxScaler()
    
    scaled_data = scaler.fit_transform(data)
    return scaled_data

# Function for clustering using KMedoids
def kmedoids_clustering(data, n_clusters, random_state=42):
    kmedoids = KMedoids(n_clusters=n_clusters, random_state=random_state)
    clusters = kmedoids.fit_predict(data)
    return clusters, kmedoids

# Function for clustering using Agglomerative Linkage
def agglomerative_clustering(data, n_clusters, linkage_method='ward', random_state=42):
    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
    clusters = clustering.fit_predict(data)
    return clusters

# Function to calculate and plot silhouette score
def calculate_silhouette_score(data, clusters):
    score = silhouette_score(data, clusters)
    return score

# Main Streamlit app
def app():
    st.title("Clustering and Silhouette Score Analysis")
    
    # File upload
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        # Load the data
        data = load_data(uploaded_file)
        st.write("Data Preview:", data.head())
        
        # Select scaling method
        scaling_method = st.selectbox("Choose scaling method", options=['standard', 'minmax'])
        
        # Apply scaling to the data
        scaled_data = scale_data(data, scaling_method)
        
        # Select clustering method
        clustering_method = st.selectbox("Choose clustering method", options=['KMedoids', 'Agglomerative Linkage'])
        
        # Number of clusters
        n_clusters = st.slider("Number of clusters", min_value=2, max_value=10, value=3)
        
        # Perform clustering based on selected method
        if clustering_method == 'KMedoids':
            clusters, _ = kmedoids_clustering(scaled_data, n_clusters)
        else:
            linkage_method = st.selectbox("Linkage method", options=['ward', 'complete', 'average', 'single'])
            clusters = agglomerative_clustering(scaled_data, n_clusters, linkage_method)
        
        # Calculate silhouette score
        silhouette_avg = calculate_silhouette_score(scaled_data, clusters)
        st.write(f"Silhouette Score: {silhouette_avg:.3f}")
        
        # Visualize clustering result
        data_with_clusters = pd.DataFrame(scaled_data, columns=data.columns)
        data_with_clusters['Cluster'] = clusters
        
        # Plot the clusters using seaborn
        fig, ax = plt.subplots()
        sns.scatterplot(x=data_with_clusters.columns[0], y=data_with_clusters.columns[1], hue='Cluster', data=data_with_clusters, palette='Set1', ax=ax)
        plt.title("Clustering Result")
        st.pyplot(fig)
        
        # Show cluster centers for KMedoids
        if clustering_method == 'KMedoids':
            st.write("Cluster Centers:")
            st.write(_.cluster_centers_)

# Run the app
if __name__ == "__main__":
    app()
