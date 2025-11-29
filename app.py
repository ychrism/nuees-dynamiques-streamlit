import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import seaborn as sns
from io import BytesIO
import base64

# Import your NueesDynamiques class
class NueesDynamiques:
    """
    Implementation of Nuées Dynamiques (Diday, 1971)
    """
    
    def __init__(self, k=3, n_etalons_per_cluster=5, max_iterations=100, 
                 tolerance=1e-4, random_state=None, aggregation_type='diday'):
        self.k = k
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.random_state = random_state
        self.aggregation_type = aggregation_type
        
        if isinstance(n_etalons_per_cluster, int):
            self.n_etalons = [n_etalons_per_cluster] * k
        else:
            self.n_etalons = n_etalons_per_cluster
            
        self.etalons = None
        self.labels = None
        self.cluster_centers = None
        self.history = []
        
    def _initialize_etalons(self, X):
        np.random.seed(self.random_state)
        etalons = []
        for ni in self.n_etalons:
            indices = np.random.choice(X.shape[0], ni, replace=False)
            etalons.append(X[indices].copy())
        return etalons
    
    def _distance_to_etalons(self, x, etalons_set):
        distances = [np.linalg.norm(x - etalon) for etalon in etalons_set]
        return np.min(distances)
    
    def _distance_to_cluster(self, x, cluster_points):
        if len(cluster_points) == 0:
            return np.inf
        center = cluster_points.mean(axis=0)
        return np.linalg.norm(x - center)
    
    def _aggregation_function(self, x, i, X, clusters, etalons):
        if self.aggregation_type == 'diday':
            d_etalons = self._distance_to_etalons(x, etalons[i])
            d_center = self._distance_to_cluster(x, clusters[i])
            d_separation = sum(self._distance_to_etalons(x, etalons[j]) 
                             for j in range(self.k) if j != i)
            if d_separation == 0:
                d_separation = 1e-10
            return (d_etalons + d_center) / d_separation
        elif self.aggregation_type == 'simple':
            return self._distance_to_cluster(x, clusters[i])
    
    def _assign_clusters(self, X, etalons):
        labels = np.zeros(X.shape[0], dtype=int)
        for idx, x in enumerate(X):
            min_dist = np.inf
            best_cluster = 0
            for i in range(self.k):
                dist = self._distance_to_etalons(x, etalons[i])
                if dist < min_dist:
                    min_dist = dist
                    best_cluster = i
            labels[idx] = best_cluster
        return labels
    
    def _update_etalons(self, X, labels, etalons):
        new_etalons = []
        clusters = [X[labels == i] for i in range(self.k)]
        
        for i in range(self.k):
            cluster_points = clusters[i]
            if len(cluster_points) == 0:
                indices = np.random.choice(X.shape[0], self.n_etalons[i], replace=False)
                new_etalons.append(X[indices].copy())
                continue
            
            r_values = []
            for x in cluster_points:
                r = self._aggregation_function(x, i, X, clusters, etalons)
                r_values.append(r)
            r_values = np.array(r_values)
            
            if len(cluster_points) >= self.n_etalons[i]:
                best_indices = np.argpartition(r_values, self.n_etalons[i])[:self.n_etalons[i]]
            else:
                best_indices = np.arange(len(cluster_points))
            new_etalons.append(cluster_points[best_indices].copy())
        return new_etalons
    
    def _calculate_cluster_centers(self, X, labels):
        centers = []
        for i in range(self.k):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                centers.append(cluster_points.mean(axis=0))
            else:
                centers.append(np.zeros(X.shape[1]))
        return np.array(centers)
    
    def _etalons_shift(self, old_etalons, new_etalons):
        total_shift = 0
        for old_set, new_set in zip(old_etalons, new_etalons):
            for old_et in old_set:
                min_dist = min(np.linalg.norm(old_et - new_et) for new_et in new_set)
                total_shift += min_dist
        return total_shift
    
    def fit(self, X):
        self.etalons = self._initialize_etalons(X)
        self.history = [{'etalons': [e.copy() for e in self.etalons], 'labels': None}]
        
        for iteration in range(self.max_iterations):
            self.labels = self._assign_clusters(X, self.etalons)
            new_etalons = self._update_etalons(X, self.labels, self.etalons)
            self.cluster_centers = self._calculate_cluster_centers(X, self.labels)
            
            self.history.append({
                'etalons': [e.copy() for e in new_etalons],
                'labels': self.labels.copy()
            })
            
            shift = self._etalons_shift(self.etalons, new_etalons)
            self.etalons = new_etalons
            
            if shift < self.tolerance:
                st.success(f"✓ Converged after {iteration + 1} iterations")
                break
        else:
            st.warning(f"Reached maximum iterations ({self.max_iterations})")
        
        return self
    
    def predict(self, X):
        return self._assign_clusters(X, self.etalons)
    
    def calculate_inertia(self, X):
        inertia = 0
        for i in range(self.k):
            cluster_points = X[self.labels == i]
            if len(cluster_points) > 0:
                center = self.cluster_centers[i]
                inertia += np.sum((cluster_points - center)**2)
        return inertia


def preprocess_data(df):
    """
    Automatically preprocess data:
    - Handle missing values
    - Scale numerical features
    """
    df_processed = df.copy()
    preprocessing_info = []
    
    # Separate numeric and categorical columns
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    #categorical_cols = df_processed.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Handle missing values
    if df_processed.isnull().sum().sum() > 0:
        # For numeric: fill with median
        for col in numeric_cols:
            if df_processed[col].isnull().sum() > 0:
                median_val = df_processed[col].median()
                df_processed[col].fillna(median_val, inplace=True)
                preprocessing_info.append(f"✓ Filled {col} missing values with median ({median_val:.2f})")
        
        # For categorical: fill with mode
        # for col in categorical_cols:
        #     if df_processed[col].isnull().sum() > 0:
        #         mode_val = df_processed[col].mode()[0]
        #         df_processed[col].fillna(mode_val, inplace=True)
        #         preprocessing_info.append(f"✓ Filled {col} missing values with mode ({mode_val})")
    
    # Encode categorical variables
    # label_encoders = {}
    # for col in categorical_cols:
    #     le = LabelEncoder()
    #     df_processed[col] = le.fit_transform(df_processed[col].astype(str))
    #     label_encoders[col] = le
    #     preprocessing_info.append(f"✓ Encoded categorical column: {col} ({len(le.classes_)} unique values)")
    
    # Get all numeric columns after encoding
    all_numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(all_numeric_cols) < 2:
        return None, None, ["❌ Error: Need at least 2 numeric columns after preprocessing"]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_processed[all_numeric_cols])
    
    preprocessing_info.append(f"✓ Standardized {len(all_numeric_cols)} features (mean=0, std=1)")
    
    return X_scaled, all_numeric_cols, preprocessing_info


def visualize_data_before_clustering(df, X_scaled, feature_names):
    """Visualize uploaded data before clustering"""
    st.subheader("📊 Data Visualization (Before Clustering)")
    
    n_features = X_scaled.shape[1]
    
    # Use PCA for visualization if more than 2 features
    if n_features > 2:
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X_scaled)
        explained_var = pca.explained_variance_ratio_
        
        col1, col2 = st.columns(2)
        
        with col1:
            # PCA scatter plot
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(X_2d[:, 0], X_2d[:, 1], alpha=0.6, s=50)
            ax.set_xlabel(f'PC1 ({explained_var[0]*100:.1f}% variance)')
            ax.set_ylabel(f'PC2 ({explained_var[1]*100:.1f}% variance)')
            ax.set_title('PCA Projection (2D)')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
            
            st.info(f"📐 Dataset has {n_features} features. Using PCA to visualize in 2D.\n\n"
                   f"PC1 + PC2 explain {(explained_var[0] + explained_var[1])*100:.1f}% of variance.")
        
        with col2:
            # Feature correlation heatmap
            fig, ax = plt.subplots(figsize=(8, 6))
            if n_features <= 10:  # Only show if reasonable number of features
                corr_matrix = pd.DataFrame(X_scaled, columns=feature_names).corr()
                sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                           center=0, ax=ax, cbar_kws={'label': 'Correlation'})
                ax.set_title('Feature Correlation Matrix')
            else:
                # Just show top correlations
                corr_matrix = pd.DataFrame(X_scaled, columns=feature_names).corr()
                sns.heatmap(corr_matrix, cmap='coolwarm', center=0, ax=ax, 
                           cbar_kws={'label': 'Correlation'})
                ax.set_title('Feature Correlation Matrix')
            st.pyplot(fig)
            plt.close()
    
    else:  # Exactly 2 features
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(X_scaled[:, 0], X_scaled[:, 1], alpha=0.6, s=50)
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])
        ax.set_title('Data Distribution (Scaled)')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()
    
    # Show basic statistics
    with st.expander("📈 Statistical Summary"):
        stats_df = df[feature_names].describe()
        st.dataframe(stats_df)


def generate_sample_data(dataset_type, n_samples=300):
    """Generate sample datasets"""
    np.random.seed(42)
    
    if dataset_type == "Blobs (round clusters)":
        from sklearn.datasets import make_blobs
        X, _ = make_blobs(n_samples=n_samples, n_features=2, centers=3, 
                         cluster_std=0.6, random_state=42)
        return X, ['Feature 1', 'Feature 2']
    
    elif dataset_type == "Elongated clusters":
        t1 = np.linspace(0, 5, n_samples // 2)
        x1 = t1 + np.random.randn(n_samples // 2) * 0.3
        y1 = np.sin(t1) + np.random.randn(n_samples // 2) * 0.3
        cluster1 = np.column_stack([x1, y1])
        
        t2 = np.linspace(0, 3, n_samples // 2)
        x2 = t2 * np.cos(t2) + np.random.randn(n_samples // 2) * 0.3 + 3
        y2 = t2 * np.sin(t2) + np.random.randn(n_samples // 2) * 0.3 + 3
        cluster2 = np.column_stack([x2, y2])
        
        return np.vstack([cluster1, cluster2]), ['Feature 1', 'Feature 2']
    
    elif dataset_type == "Moons":
        from sklearn.datasets import make_moons
        X, _ = make_moons(n_samples=n_samples, noise=0.1, random_state=42)
        return X, ['Feature 1', 'Feature 2']
    
    elif dataset_type == "Circles":
        from sklearn.datasets import make_circles
        X, _ = make_circles(n_samples=n_samples, noise=0.05, factor=0.5, random_state=42)
        return X, ['Feature 1', 'Feature 2']
    
    elif dataset_type == "High-dimensional (10 features)":
        from sklearn.datasets import make_blobs
        X, _ = make_blobs(n_samples=n_samples, n_features=10, centers=4, 
                         cluster_std=1.5, random_state=42)
        return X, [f'Feature {i+1}' for i in range(10)]


def plot_clustering(X, model, title="Nuées Dynamiques Clustering", feature_names=None):
    """Create clustering visualization"""
    n_features = X.shape[1]
    
    # Use PCA if more than 2 dimensions
    if n_features > 2:
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
        explained_var = pca.explained_variance_ratio_
        xlabel = f'PC1 ({explained_var[0]*100:.1f}% var)'
        ylabel = f'PC2 ({explained_var[1]*100:.1f}% var)'
        
        # Project étalons and centers to 2D
        etalons_2d = []
        for etalons_set in model.etalons:
            etalons_2d.append(pca.transform(etalons_set))
        centers_2d = pca.transform(model.cluster_centers)
    else:
        X_2d = X
        xlabel = feature_names[0] if feature_names else 'Feature 1'
        ylabel = feature_names[1] if feature_names else 'Feature 2'
        etalons_2d = model.etalons
        centers_2d = model.cluster_centers
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = plt.cm.viridis(np.linspace(0, 1, model.k))
    
    # Plot data points
    for i in range(model.k):
        cluster_mask = model.labels == i
        cluster_points_2d = X_2d[cluster_mask]
        ax.scatter(cluster_points_2d[:, 0], cluster_points_2d[:, 1], 
                  c=[colors[i]], alpha=0.5, s=50, label=f'Cluster {i+1}')
        
        # Plot étalons
        etalons = etalons_2d[i]
        ax.scatter(etalons[:, 0], etalons[:, 1], 
                  c=[colors[i]], marker='s', s=150, 
                  edgecolors='black', linewidths=2)
        
        # Connect étalons to show skeleton
        if len(etalons) > 1:
            sorted_indices = np.argsort(etalons[:, 0])
            sorted_etalons = etalons[sorted_indices]
            ax.plot(sorted_etalons[:, 0], sorted_etalons[:, 1], 
                   'k--', alpha=0.3, linewidth=1)
        
        # Plot cluster center
        ax.scatter(centers_2d[i, 0], centers_2d[i, 1],
                  c='red', marker='X', s=200, edgecolors='black', linewidths=2)
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig


# Streamlit App
def main():
    st.set_page_config(page_title="Nuées Dynamiques", page_icon="🔬", layout="wide")
    
    st.title("🔬 Nuées Dynamiques (Dynamic Clouds) Clustering")
    st.markdown("""
    **Nuées Dynamiques** (Diday, 1971) is an advanced clustering algorithm that uses 
    multiple representatives (étalons) per cluster instead of a single centroid, 
    allowing it to better capture complex cluster shapes.
    """)
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Data source selection
    data_source = st.sidebar.radio(
        "Data Source",
        ["Upload CSV", "Generate Sample Data"]
    )
    
    X = None
    feature_names = None
    original_df = None
    
    if data_source == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                original_df = df.copy()
                
                st.subheader("📋 Uploaded Data Preview")
                st.dataframe(df.head(10), use_container_width=True)
                
                st.info(f"📊 Dataset shape: **{df.shape[0]} rows × {df.shape[1]} columns**")
                
                # Show column types
                with st.expander("📑 Column Information"):
                    col_info = pd.DataFrame({
                        'Column': df.columns,
                        'Type': df.dtypes.astype(str),
                        'Non-Null': df.notnull().sum(),
                        'Null': df.isnull().sum(),
                        'Unique': df.nunique()
                    })
                    st.dataframe(col_info, use_container_width=True)
                
                # Preprocess data
                st.subheader("🔧 Automatic Preprocessing")
                with st.spinner("Processing data..."):
                    X, feature_names, preprocessing_info = preprocess_data(df)
                
                if X is None:
                    for info in preprocessing_info:
                        st.error(info)
                    return
                
                # Show preprocessing steps
                for info in preprocessing_info:
                    st.success(info)
                
                st.info(f"✅ Ready for clustering with **{X.shape[1]} features** and **{X.shape[0]} samples**")
                
                # Visualize data before clustering
                visualize_data_before_clustering(df, X, feature_names)
                
            except Exception as e:
                st.error(f"Error loading CSV: {str(e)}")
                return
            
    else:  # Generate sample data
        dataset_type = st.sidebar.selectbox(
            "Select Dataset Type",
            ["Blobs (round clusters)", "Elongated clusters", "Moons", "Circles", "High-dimensional (10 features)"]
        )
        
        n_samples = st.sidebar.slider("Number of samples", 100, 1000, 300, 50)
        X, feature_names = generate_sample_data(dataset_type, n_samples)
        
        st.subheader(f"📊 Generated Data: {dataset_type}")
        st.info(f"Dataset shape: **{X.shape[0]} samples × {X.shape[1]} features**")
        
        # Show raw data visualization
        if X.shape[1] == 2:
            fig_raw, ax_raw = plt.subplots(figsize=(10, 6))
            ax_raw.scatter(X[:, 0], X[:, 1], alpha=0.6, s=50)
            ax_raw.set_xlabel(feature_names[0])
            ax_raw.set_ylabel(feature_names[1])
            ax_raw.set_title('Raw Data (before clustering)')
            ax_raw.grid(True, alpha=0.3)
            st.pyplot(fig_raw)
            plt.close()
        else:
            # Use PCA for visualization
            pca = PCA(n_components=2)
            X_2d = pca.fit_transform(X)
            explained_var = pca.explained_variance_ratio_
            
            fig_raw, ax_raw = plt.subplots(figsize=(10, 6))
            ax_raw.scatter(X_2d[:, 0], X_2d[:, 1], alpha=0.6, s=50)
            ax_raw.set_xlabel(f'PC1 ({explained_var[0]*100:.1f}% variance)')
            ax_raw.set_ylabel(f'PC2 ({explained_var[1]*100:.1f}% variance)')
            ax_raw.set_title(f'Raw Data - PCA Projection ({(explained_var[0] + explained_var[1])*100:.1f}% variance explained)')
            ax_raw.grid(True, alpha=0.3)
            st.pyplot(fig_raw)
            plt.close()
    
    if X is not None:
        st.sidebar.markdown("---")
        st.sidebar.header("🎯 Algorithm Parameters")
        
        k = st.sidebar.slider("Number of clusters (K)", 2, 10, 3, 1)
        n_etalons = st.sidebar.slider("Étalons per cluster", 1, 20, 5, 1)
        max_iterations = st.sidebar.slider("Max iterations", 10, 200, 100, 10)
        aggregation_type = st.sidebar.selectbox(
            "Aggregation function",
            ["diday", "simple"],
            help="'diday' uses Diday's aggregation-separation function, 'simple' uses distance to center"
        )
        random_state = st.sidebar.number_input("Random seed", 0, 1000, 42, 1)
        
        # Run clustering button
        if st.sidebar.button("🚀 Run Clustering", type="primary"):
            with st.spinner("Running Nuées Dynamiques algorithm..."):
                # Create and fit model
                model = NueesDynamiques(
                    k=k,
                    n_etalons_per_cluster=n_etalons,
                    max_iterations=max_iterations,
                    tolerance=1e-4,
                    random_state=random_state,
                    aggregation_type=aggregation_type
                )
                
                model.fit(X)
                
                # Store results in session state
                st.session_state['model'] = model
                st.session_state['X'] = X
                st.session_state['feature_names'] = feature_names
                st.session_state['original_df'] = original_df
        
        # Display results if model exists
        if 'model' in st.session_state:
            model = st.session_state['model']
            X = st.session_state['X']
            feature_names = st.session_state['feature_names']
            original_df = st.session_state.get('original_df', None)
            
            st.markdown("---")
            st.header("📈 Results")
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Number of Clusters", model.k)
            
            with col2:
                inertia = model.calculate_inertia(X)
                st.metric("Inertia", f"{inertia:.2f}")
            
            with col3:
                st.metric("Iterations", len(model.history) - 1)
            
            with col4:
                st.metric("Features Used", X.shape[1])
            
            # Cluster sizes
            st.subheader("📊 Cluster Statistics")
            cluster_stats = []
            for i in range(model.k):
                size = np.sum(model.labels == i)
                n_etalons_actual = len(model.etalons[i])
                cluster_stats.append({
                    'Cluster': i + 1,
                    'Size': size,
                    'Percentage': f"{100 * size / len(X):.1f}%",
                    'Étalons': n_etalons_actual
                })
            
            st.dataframe(pd.DataFrame(cluster_stats), use_container_width=True)
            
            # Visualization
            st.subheader("🎨 Visualization")
            fig = plot_clustering(X, model, "Nuées Dynamiques Clustering Results", feature_names)
            st.pyplot(fig)
            plt.close()
            
            if X.shape[1] > 2:
                st.info("💡 **Note**: Data has more than 2 features. Visualization uses PCA projection to 2D.")
            
            st.info("💡 **Legend**: Colored points = data, Squares = étalons (representatives), Red X = cluster centers, Dashed lines = étalon skeleton")
            
            # Download results
            st.subheader("💾 Download Results")
            
            # Create results dataframe
            if original_df is not None:
                results_df = original_df.copy()
            else:
                if feature_names and len(feature_names) == X.shape[1]:
                    results_df = pd.DataFrame(X, columns=feature_names)
                else:
                    results_df = pd.DataFrame(X, columns=[f'Feature_{i+1}' for i in range(X.shape[1])])
            
            results_df['Cluster'] = model.labels + 1  # 1-indexed for users
            
            st.download_button(
                label="📥 Download Clustered Data (CSV)",
                data=results_df.to_csv(index=False),
                file_name="nuees_dynamiques_results.csv",
                mime="text/csv"
            )
            
            # Show sample of results
            with st.expander("👀 Preview Results"):
                st.dataframe(results_df.head(20), use_container_width=True)
            
            # Additional information
            with st.expander("ℹ️ About Nuées Dynamiques"):
                st.markdown("""
                ### What is Nuées Dynamiques?
                
                Nuées Dynamiques (Dynamic Clouds) is a clustering algorithm introduced by Édouard Diday in 1971.
                
                **Key Differences from K-means:**
                - **K-means**: Uses 1 centroid per cluster
                - **Nuées Dynamiques**: Uses multiple étalons (representatives) per cluster
                
                **Advantages:**
                - Better captures elongated or complex cluster shapes
                - Étalons form a "skeleton" that represents the cluster structure
                - More robust for non-spherical clusters
                - Works with high-dimensional data (automatic PCA visualization)
                
                **The Algorithm:**
                1. Initialize random étalons for each cluster
                2. Assign points to nearest étalon
                3. Update étalons by selecting best representatives from each cluster
                4. Repeat until convergence
                
                **Automatic Preprocessing:**
                - Categorical variables are automatically encoded
                - Missing values are imputed (median for numeric, mode for categorical)
                - All features are standardized (mean=0, std=1)
                
                **Reference:** Diday, E. (1971). Une nouvelle méthode en classification automatique 
                et reconnaissance des formes: la méthode des nuées dynamiques. 
                *Revue de Statistique Appliquée*, 19(2), 19-33.
                """)


if __name__ == "__main__":
    main()