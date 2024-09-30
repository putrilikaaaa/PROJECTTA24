def pemetaan():
    st.subheader("Pemetaan Clustering dengan DTW")
    data_df = upload_csv_file(key="pemetaan_upload")

    if data_df is not None:
        data_df['Tanggal'] = pd.to_datetime(data_df['Tanggal'], format='%d-%b-%y', errors='coerce')
        data_df.set_index('Tanggal', inplace=True)

        # Calculate daily averages
        data_daily = data_df.resample('D').mean()

        # Standardize daily data
        scaler = StandardScaler()
        data_daily_standardized = pd.DataFrame(scaler.fit_transform(data_daily), index=data_daily.index, columns=data_daily.columns)

        # Compute local cost matrix and accumulated cost matrix
        local_cost_matrix_daily = compute_local_cost_matrix(data_daily_standardized)
        accumulated_cost_matrix_daily = compute_accumulated_cost_matrix(local_cost_matrix_daily)

        # Compute DTW distance matrix for daily data
        dtw_distance_matrix_daily = compute_dtw_distance_matrix(accumulated_cost_matrix_daily)

        # Clustering and silhouette score calculation for daily data
        max_n_clusters = 10
        silhouette_scores = {}

        for n_clusters in range(2, max_n_clusters + 1):
            # Agglomerative Clustering
            clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='complete')
            labels = clustering.fit_predict(dtw_distance_matrix_daily)
            score = silhouette_score(dtw_distance_matrix_daily, labels, metric='precomputed')
            silhouette_scores[n_clusters] = score

            # K-Medoids clustering
            try:
                kmedoids = KMedoids(n_clusters=n_clusters, metric='precomputed')
                kmedoids_labels = kmedoids.fit_predict(dtw_distance_matrix_daily)
                kmedoids_score = silhouette_score(dtw_distance_matrix_daily, kmedoids_labels, metric='precomputed')
                silhouette_scores[f'KMedoids-{n_clusters}'] = kmedoids_score
            except Exception as e:
                st.warning(f"Error occurred while performing K-Medoids for {n_clusters} clusters: {e}")

        # Plot Silhouette Scores
        plt.figure(figsize=(10, 6))
        plt.plot(list(silhouette_scores.keys()), list(silhouette_scores.values()), marker='o', linestyle='-')
        plt.title('Silhouette Score vs. Number of Clusters (Data Harian)')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')
        plt.xticks(rotation=45)
        plt.grid(True)
        st.pyplot(plt)

        # Determine optimal number of clusters
        optimal_n_clusters = None
        try:
            optimal_n_clusters = max((k for k in silhouette_scores.keys() if k.startswith('KMedoids-')), key=silhouette_scores.get)
        except ValueError:
            st.warning("Tidak ada hasil untuk K-Medoids. Memilih jumlah kluster berdasarkan Agglomerative Clustering.")
            optimal_n_clusters = max(silhouette_scores.keys(), key=silhouette_scores.get)

        optimal_n_clusters_num = int(optimal_n_clusters.split('-')[1]) if 'KMedoids-' in optimal_n_clusters else optimal_n_clusters
        st.write(f"Jumlah kluster optimal berdasarkan Silhouette Score adalah: {optimal_n_clusters_num}")

        # Clustering and dendrogram
        condensed_dtw_distance_matrix = squareform(dtw_distance_matrix_daily)
        Z = linkage(condensed_dtw_distance_matrix, method='complete')

        plt.figure(figsize=(16, 10))
        dendrogram(Z, labels=data_daily_standardized.columns, leaf_rotation=90)
        plt.title('Dendrogram Clustering dengan DTW (Data Harian)')
        plt.xlabel('Provinsi')
        plt.ylabel('Jarak DTW')
        st.pyplot(plt)

        # Table of provinces per cluster for Agglomerative Clustering
        cluster_labels = AgglomerativeClustering(n_clusters=optimal_n_clusters_num, metric='precomputed', linkage='complete').fit_predict(dtw_distance_matrix_daily)
        clustered_data = pd.DataFrame({
            'Province': data_daily_standardized.columns,
            'Cluster': cluster_labels
        })

        # Display cluster table
        st.subheader("Tabel Provinsi per Cluster (Agglomerative)")
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
                2: 'green'
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
            gdf.plot(ax=ax, color=gdf['color'], edgecolor='black', alpha=0.6)  # Plot provinces with colors

            # Add title and labels
            plt.title('Peta Kluster Provinsi di Indonesia', fontsize=15)
            plt.xlabel('Longitude', fontsize=12)
            plt.ylabel('Latitude', fontsize=12)
            st.pyplot(plt)
        else:
            st.warning("Silakan upload file GeoJSON.")
