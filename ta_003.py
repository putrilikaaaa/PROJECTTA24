import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import geopandas as gpd

# Assuming `data_df` is uploaded, replace with the actual data loading mechanism
def load_data():
    # Placeholder for loading the data
    # You can modify this to your actual data loading process
    return pd.read_csv("your_data.csv")  # Replace with actual file or method

# Compute the DTW distance matrix for the given data
def compute_local_cost_matrix(df):
    # Add your DTW computation here
    return np.random.rand(df.shape[0], df.shape[0])  # Placeholder, replace with actual DTW computation

def compute_accumulated_cost_matrix(local_cost_matrix):
    # Placeholder, implement your actual accumulated cost calculation
    return local_cost_matrix

def compute_dtw_distance_matrix(accumulated_cost_matrix):
    # Placeholder for DTW matrix calculation, you can adjust this with your actual DTW logic
    return np.random.rand(accumulated_cost_matrix.shape[0], accumulated_cost_matrix.shape[0])

def symmetrize(matrix):
    return (matrix + matrix.T) / 2

def plot_dendrogram(linkage_matrix):
    plt.figure(figsize=(10, 7))
    dendrogram(linkage_matrix)
    plt.show()

# Function to calculate silhouette score
def silhouette_analysis(dtw_distance_matrix, max_n_clusters=10):
    silhouette_scores = {}
    cluster_labels_dict = {}
    condensed_dtw_distance_matrix = squareform(dtw_distance_matrix)

    # Iterate over different numbers of clusters to find the best silhouette score
    for n_clusters in range(2, max_n_clusters + 1):
        clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='complete')
        labels = clustering.fit_predict(condensed_dtw_distance_matrix)

        # Ensure labels and distance matrix have matching sizes
        if len(labels) == condensed_dtw_distance_matrix.shape[0]:
            score = silhouette_score(condensed_dtw_distance_matrix, labels, metric='precomputed')
            silhouette_scores[n_clusters] = score
            cluster_labels_dict[n_clusters] = labels
        else:
            st.warning(f"Number of labels ({len(labels)}) does not match the number of samples ({condensed_dtw_distance_matrix.shape[0]}).")
    
    return silhouette_scores, cluster_labels_dict

# Pemetaan function to perform the clustering and display relevant charts
def pemetaan(data_df):
    st.title("Pemetaan Clustering")

    # Process daily data (adjust according to your input format)
    data_daily = data_df.set_index('Tanggal')  # Ensure you have a 'Tanggal' column
    data_daily_values = data_daily.values

    # Compute DTW distance matrix for daily data
    local_cost_matrix_daily = compute_local_cost_matrix(pd.DataFrame(data_daily_values, columns=data_daily.columns))
    accumulated_cost_matrix_daily = compute_accumulated_cost_matrix(local_cost_matrix_daily)
    dtw_distance_matrix_daily = compute_dtw_distance_matrix(accumulated_cost_matrix_daily)

    # Ensure symmetry of the DTW distance matrix
    dtw_distance_matrix_daily = symmetrize(dtw_distance_matrix_daily)

    # Perform silhouette analysis for different numbers of clusters
    silhouette_scores, cluster_labels_dict = silhouette_analysis(dtw_distance_matrix_daily)

    # Display silhouette scores for different cluster numbers
    st.write("Silhouette Scores per Cluster Count")
    for n_clusters, score in silhouette_scores.items():
        st.write(f"{n_clusters} clusters: {score:.4f}")

    # Plot the silhouette scores
    fig, ax = plt.subplots()
    sns.lineplot(x=list(silhouette_scores.keys()), y=list(silhouette_scores.values()), ax=ax)
    ax.set_title("Silhouette Scores for Different Numbers of Clusters")
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("Silhouette Score")
    st.pyplot(fig)

    # Select the best clustering solution (based on highest silhouette score)
    best_n_clusters = max(silhouette_scores, key=silhouette_scores.get)
    st.write(f"Best number of clusters: {best_n_clusters}")

    # Display the clusters
    labels = cluster_labels_dict[best_n_clusters]
    st.write(f"Cluster Labels: {labels}")

    # Add GeoJSON and clustering map (assuming 'Propinsi' column for provinces)
    geojson_file = 'indonesia-prov.geojson'
    gdf = gpd.read_file(geojson_file)
    
    # Map the labels to the provinces
    gdf['Cluster'] = labels
    
    # Plot the map
    st.write("Cluster Visualization on Indonesian Map")
    fig, ax = plt.subplots(figsize=(10, 10))
    gdf.plot(column='Cluster', ax=ax, legend=True, cmap='RdYlGn')
    plt.title("Cluster Map")
    st.pyplot(fig)

# Main function to run the app
def main():
    st.title("Clustering and Pemetaan Application")
    
    # Upload data
    uploaded_file = st.file_uploader("Upload your data CSV", type=["csv"])
    
    if uploaded_file:
        data_df = pd.read_csv(uploaded_file)
        
        # Ensure data is valid
        if 'Tanggal' not in data_df.columns:
            st.error("Data must contain a 'Tanggal' column")
        else:
            pemetaan(data_df)

if __name__ == "__main__":
    main()
