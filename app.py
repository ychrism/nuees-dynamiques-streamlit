import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.metrics import silhouette_score, davies_bouldin_score
import seaborn as sns
from matplotlib.patches import Ellipse, Rectangle
import warnings
import time
warnings.filterwarnings('ignore')

# Configuration de style
plt.style.use('seaborn-v0_8-darkgrid')

# ============ CLASSE NueesDynamiques ============
class NueesDynamiques:
    """
    Implémentation de l'algorithme des Nuées Dynamiques selon Diday (1971).
    
    Cet algorithme utilise deux fonctions principales:
    - φ (phi): Fonction de réallocation - affecte chaque individu au noyau le plus proche
    - ψ (psi): Fonction de recentrage - recalcule les noyaux à partir des classes formées
    
    La distance d'un point à un ensemble (noyau) est définie comme la distance du point
    au centre de gravité (centre) de cet ensemble: d(x, E_i) = d(x, G_i)
    """
    
    def __init__(self, k=3, n_etalons_per_cluster=5, max_iterations=100, 
                 tolerance=1e-4, random_state=None, representation_type='etalons',
                 aggregation_type='diday', distribution_type='gaussian'):
        self.k = k
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.random_state = random_state
        self.representation_type = representation_type
        self.aggregation_type = aggregation_type
        self.distribution_type = distribution_type
        
        if representation_type == 'singleton':
            n_etalons_per_cluster = 1
        
        self.n_etalons = [n_etalons_per_cluster] * k if isinstance(n_etalons_per_cluster, int) else n_etalons_per_cluster
        self.etalons = None
        self.labels = None
        self.cluster_centers = None  # Centres de gravité G_i
        self.cluster_axes = None
        self.cluster_distributions = None
        self.history = []
        self.execution_time = 0
        self.silhouette_score = None
        self.davies_bouldin_score = None
        self.intra_class_variance = []  # Historique de la variance intra-classes
    
    def _initialize_representation(self, X):
        """Initialise les noyaux (représentations) des clusters."""
        np.random.seed(self.random_state)
        if self.representation_type == 'etalons':
            return [X[np.random.choice(X.shape[0], ni, replace=False)].copy() for ni in self.n_etalons]
        elif self.representation_type == 'singleton':
            indices = np.random.choice(X.shape[0], self.k, replace=False)
            return [X[idx:idx+1] for idx in indices]
        elif self.representation_type == 'axis':
            return self._initialize_axes(X)
        elif self.representation_type == 'distribution':
            return self._initialize_distributions(X)
    
    def _initialize_axes(self, X):
        """Initialise les axes principaux pour chaque cluster."""
        axes = []
        for i in range(self.k):
            indices = np.random.choice(X.shape[0], min(100, X.shape[0] // self.k), replace=False)
            cluster_sample = X[indices]
            center = cluster_sample.mean(axis=0)
            if len(cluster_sample) > 1:
                cov_matrix = np.cov(cluster_sample.T)
                eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
                main_axis = eigenvectors[:, np.argmax(eigenvalues)]
            else:
                main_axis = np.ones(X.shape[1]) / np.sqrt(X.shape[1])
            axes.append({'center': center, 'direction': main_axis})
        return axes
    
    def _initialize_distributions(self, X):
        """Initialise les distributions (gaussienne ou uniforme) pour chaque cluster."""
        distributions = []
        for i in range(self.k):
            indices = np.random.choice(X.shape[0], min(50, X.shape[0] // self.k), replace=False)
            cluster_sample = X[indices]
            mean = cluster_sample.mean(axis=0)
            cov = np.cov(cluster_sample.T) + np.eye(X.shape[1]) * 1e-6 if len(cluster_sample) > 1 else np.eye(X.shape[1])
            
            if self.distribution_type == 'gaussian':
                distributions.append({'mean': mean, 'cov': cov, 'type': 'gaussian'})
            else:
                distributions.append({'min': cluster_sample.min(axis=0), 'max': cluster_sample.max(axis=0), 'type': 'uniform'})
        return distributions
    
    def _calculate_cluster_centers(self, X, labels):
        """
        Calcule les centres de gravité G_i pour chaque cluster.
        
        G_i = (1/n_i) * Σ(x_j) pour tous x_j dans le cluster i
        
        où n_i est le nombre d'éléments dans le cluster i.
        """
        centers = np.zeros((self.k, X.shape[1]))
        for i in range(self.k):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                centers[i] = cluster_points.mean(axis=0)
            else:
                centers[i] = np.zeros(X.shape[1])
        return centers
    
    def _distance_to_representation(self, x, cluster_center):
        """
        Calcule la distance d'un point x à un cluster via son centre de gravité.
        
        Selon Diday (1971), la distance d'un point à un ensemble (noyau) est définie comme
        la distance du point au centre de gravité (centre) de cet ensemble:
        
        d(x, E_i) = d(x, G_i) = ||x - G_i||
        
        où G_i est le centre de gravité du noyau E_i.
        
        Args:
            x: Point individu
            cluster_center: Centre de gravité G_i du cluster
        
        Returns:
            Distance euclidienne du point au centre du cluster
        """
        return np.linalg.norm(x - cluster_center)
    
    def _assign_clusters(self, X, cluster_centers):
        """
        Fonction φ (phi) - Réallocation.
        
        Affecte chaque individu x_i au cluster dont le centre de gravité est le plus proche.
        
        C_i = {x_i | x_i ∈ E et d(x_i, G_i) ≤ d(x_i, G_j) ∀j ≠ i}
        
        où G_i est le centre de gravité du cluster i.
        """
        labels = np.zeros(X.shape[0], dtype=int)
        for idx, x in enumerate(X):
            distances = [self._distance_to_representation(x, cluster_centers[i]) for i in range(self.k)]
            labels[idx] = np.argmin(distances)
        return labels
    
    def _update_representation(self, X, labels, representation):
        """
        Fonction ψ (psi) - Recentrage.
        
        Recalcule les noyaux (représentations) à partir des classes formées.
        Chaque noyau E_i est constitué d'un ensemble d'étalons qui minimisent
        une fonction de dissemblance.
        """
        clusters = [X[labels == i] for i in range(self.k)]
        
        if self.representation_type == 'etalons':
            return self._update_etalons(X, labels, representation, clusters)
        elif self.representation_type == 'singleton':
            return [[cluster.mean(axis=0)] if len(cluster) > 0 else [np.zeros(X.shape[1])] for cluster in clusters]
        elif self.representation_type == 'axis':
            return self._update_axes(clusters)
        else:
            return self._update_distributions(clusters)
    
    def _update_etalons(self, X, labels, etalons, clusters):
        """
        Met à jour les étalons (points représentatifs) pour chaque cluster.
        
        Les étalons sont sélectionnés comme les points du cluster qui minimisent
        la fonction d'agrégation-écartement.
        """
        new_etalons = []
        for i in range(self.k):
            if len(clusters[i]) == 0:
                indices = np.random.choice(X.shape[0], self.n_etalons[i], replace=False)
                new_etalons.append(X[indices].copy())
                continue
            
            # Sélectionner les n_etalons[i] points les plus centraux du cluster
            center = clusters[i].mean(axis=0)
            distances_to_center = np.array([np.linalg.norm(x - center) for x in clusters[i]])
            n_select = min(self.n_etalons[i], len(clusters[i]))
            best_indices = np.argsort(distances_to_center)[:n_select]
            new_etalons.append(clusters[i][best_indices].copy())
        return new_etalons
    
    def _update_axes(self, clusters):
        """Met à jour les axes principaux pour chaque cluster."""
        new_axes = []
        for cluster in clusters:
            if len(cluster) > 1:
                center = cluster.mean(axis=0)
                cov_matrix = np.cov(cluster.T)
                eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
                main_axis = eigenvectors[:, np.argmax(eigenvalues)]
                new_axes.append({'center': center, 'direction': main_axis})
            elif len(cluster) == 1:
                new_axes.append({'center': cluster[0], 'direction': np.ones(cluster.shape[1])/np.sqrt(cluster.shape[1])})
            else:
                new_axes.append({'center': np.zeros(clusters[0].shape[1]), 'direction': np.ones(clusters[0].shape[1])/np.sqrt(clusters[0].shape[1])})
        return new_axes
    
    def _update_distributions(self, clusters):
        """Met à jour les distributions (gaussienne ou uniforme) pour chaque cluster."""
        new_distributions = []
        for cluster in clusters:
            if len(cluster) > 0:
                mean = cluster.mean(axis=0)
                cov = np.cov(cluster.T) + np.eye(cluster.shape[1]) * 1e-6 if len(cluster) > 1 else np.eye(cluster.shape[1])
                
                if self.distribution_type == 'gaussian':
                    new_distributions.append({'mean': mean, 'cov': cov, 'type': 'gaussian'})
                else:
                    new_distributions.append({'min': cluster.min(axis=0), 'max': cluster.max(axis=0), 'type': 'uniform'})
            else:
                if self.distribution_type == 'gaussian':
                    new_distributions.append({'mean': np.zeros(clusters[0].shape[1]), 'cov': np.eye(clusters[0].shape[1]), 'type': 'gaussian'})
                else:
                    new_distributions.append({'min': np.zeros(clusters[0].shape[1]), 'max': np.ones(clusters[0].shape[1]), 'type': 'uniform'})
        return new_distributions
    
    def _calculate_intra_class_variance(self, X, labels, cluster_centers):
        """
        Calcule la variance intra-classes (critère de convergence).
        
        V(t) = Σ_i { Σ_{x_j ∈ C_i} d²(x_j, G_i) }
        
        où:
        - C_i est le cluster i
        - G_i est le centre de gravité du cluster i
        - d²(x_j, G_i) est le carré de la distance euclidienne
        
        Cette variance doit décroître (ou rester stationnaire) à chaque itération.
        """
        variance = 0
        for i in range(self.k):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                center = cluster_centers[i]
                distances_squared = np.sum((cluster_points - center) ** 2, axis=1)
                variance += np.sum(distances_squared)
        return variance
    
    def _representation_shift(self, old_repr, new_repr):
        """Calcule le changement entre deux représentations successives."""
        total_shift = 0
        if self.representation_type in ['etalons', 'singleton']:
            for old_set, new_set in zip(old_repr, new_repr):
                for old_item in old_set:
                    min_dist = min(np.linalg.norm(old_item - new_item) for new_item in new_set) if len(new_set) > 0 else np.inf
                    total_shift += min_dist
        elif self.representation_type == 'axis':
            for old_ax, new_ax in zip(old_repr, new_repr):
                total_shift += np.linalg.norm(old_ax['center'] - new_ax['center']) + (1 - np.abs(np.dot(old_ax['direction'], new_ax['direction'])))
        elif self.representation_type == 'distribution':
            for old_d, new_d in zip(old_repr, new_repr):
                if old_d['type'] == new_d['type'] == 'gaussian':
                    total_shift += np.linalg.norm(old_d['mean'] - new_d['mean']) + 0.1 * np.linalg.norm(old_d['cov'] - new_d['cov'])
        return total_shift
    
    def fit(self, X):
        """
        Entraîne le modèle des Nuées Dynamiques.
        
        Algorithme:
        1. Initialisation: Choisir k noyaux initiaux
        2. Itération:
           a. Fonction φ: Affecter chaque point au noyau le plus proche
           b. Fonction ψ: Recalculer les noyaux à partir des classes formées
           c. Vérifier la convergence (variance intra-classes)
        3. Arrêt: Quand la variance ne décroît plus ou max itérations atteint
        """
        start_time = time.time()
        
        self.representation = self._initialize_representation(X)
        self.cluster_centers = self._calculate_cluster_centers(X, np.zeros(X.shape[0], dtype=int))
        self.history = [{'representation': self.representation.copy(), 'labels': None}]
        
        for iteration in range(self.max_iterations):
            # Fonction φ: Réallocation - affecter chaque point au cluster le plus proche
            self.labels = self._assign_clusters(X, self.cluster_centers)
            
            # Calculer les nouveaux centres de gravité
            self.cluster_centers = self._calculate_cluster_centers(X, self.labels)
            
            # Fonction ψ: Recentrage - recalculer les noyaux
            new_representation = self._update_representation(X, self.labels, self.representation)
            
            # Mettre à jour les représentations spécifiques
            if self.representation_type == 'etalons':
                self.etalons = new_representation
            elif self.representation_type == 'axis':
                self.cluster_axes = new_representation
            elif self.representation_type == 'distribution':
                self.cluster_distributions = new_representation
            
            # Calculer la variance intra-classes pour le critère de convergence
            intra_variance = self._calculate_intra_class_variance(X, self.labels, self.cluster_centers)
            self.intra_class_variance.append(intra_variance)
            
            self.history.append({'representation': new_representation.copy(), 'labels': self.labels.copy()})
            
            # Vérifier la convergence
            shift = self._representation_shift(self.representation, new_representation)
            self.representation = new_representation
            
            if shift < self.tolerance:
                break
        
        self.execution_time = (time.time() - start_time) * 1000  # en millisecondes
        
        # Calculer les métriques
        try:
            self.silhouette_score = silhouette_score(X, self.labels)
        except:
            self.silhouette_score = None
        
        try:
            self.davies_bouldin_score = davies_bouldin_score(X, self.labels)
        except:
            self.davies_bouldin_score = None
        
        return self
    
    def predict(self, X):
        """Prédit les labels pour de nouvelles données."""
        return self._assign_clusters(X, self.cluster_centers)
    
    def calculate_inertia(self, X):
        """
        Calcule l'inertie (somme des carrés intra-cluster).
        
        Inertia = Σ_i { Σ_{x_j ∈ C_i} ||x_j - G_i||² }
        """
        if len(self.labels) != X.shape[0]:
            raise ValueError(f"Dimension mismatch: labels length ({len(self.labels)}) != X samples ({X.shape[0]})")
        
        inertia = 0
        for i in range(self.k):
            if np.sum(self.labels == i) > 0:
                cluster_points = X[self.labels == i]
                if len(cluster_points) > 0:
                    center = self.cluster_centers[i]
                    inertia += np.sum((cluster_points - center)**2)
        return inertia
    
    def get_representation_info(self):
        """Retourne des informations sur la représentation utilisée."""
        if self.representation_type == 'etalons':
            return [f"Cluster {i+1}: {len(self.etalons[i])} étalons" for i in range(self.k)]
        elif self.representation_type == 'singleton':
            return [f"Cluster {i+1}: Single centroid" for i in range(self.k)]
        elif self.representation_type == 'axis' and self.cluster_axes:
            return [f"Cluster {i+1}: Principal axis (norm: {np.linalg.norm(self.cluster_axes[i]['direction']):.3f})" for i in range(self.k)]
        elif self.representation_type == 'distribution' and self.cluster_distributions:
            if self.distribution_type == 'gaussian':
                return [f"Cluster {i+1}: Gaussian (det(cov): {np.linalg.det(self.cluster_distributions[i]['cov']):.3e})" for i in range(self.k)]
            return [f"Cluster {i+1}: Uniform distribution" for i in range(self.k)]
        return []


# ============ FONCTIONS UTILITAIRES ============
def preprocess_data(df):
    """Prétraite les données: imputation et normalisation."""
    df_processed = df.copy()
    preprocessing_info = []
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    
    if df_processed.isnull().sum().sum() > 0:
        for col in numeric_cols:
            if df_processed[col].isnull().sum() > 0:
                median_val = df_processed[col].median()
                df_processed[col].fillna(median_val, inplace=True)
                preprocessing_info.append(f"✓ {col}: missing values filled with median ({median_val:.2f})")
    
    all_numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(all_numeric_cols) < 2:
        return None, None, ["❌ Need at least 2 numeric columns"]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_processed[all_numeric_cols])
    preprocessing_info.append(f"✓ Standardized {len(all_numeric_cols)} features")
    
    return X_scaled, all_numeric_cols, preprocessing_info


def generate_sample_data(dataset_type, n_samples=300):
    """Génère des données synthétiques pour démonstration."""
    np.random.seed(42)
    
    if dataset_type == "Blobs":
        X, _ = make_blobs(n_samples=n_samples, n_features=2, centers=3, cluster_std=0.6, random_state=42)
    elif dataset_type == "Elongated":
        t1 = np.linspace(0, 5, n_samples // 2)
        cluster1 = np.column_stack([t1 + np.random.randn(n_samples // 2) * 0.3, np.sin(t1) + np.random.randn(n_samples // 2) * 0.3])
        t2 = np.linspace(0, 3, n_samples // 2)
        cluster2 = np.column_stack([t2 * np.cos(t2) + np.random.randn(n_samples // 2) * 0.3 + 3, 
                                    t2 * np.sin(t2) + np.random.randn(n_samples // 2) * 0.3 + 3])
        X = np.vstack([cluster1, cluster2])
    elif dataset_type == "Moons":
        X, _ = make_moons(n_samples=n_samples, noise=0.1, random_state=42)
    elif dataset_type == "Circles":
        X, _ = make_circles(n_samples=n_samples, noise=0.05, factor=0.5, random_state=42)
    else:  # High-dimensional
        X, _ = make_blobs(n_samples=n_samples, n_features=10, centers=4, cluster_std=1.5, random_state=42)
        return X, [f'F{i+1}' for i in range(10)]
    
    return X, ['Feature 1', 'Feature 2']


def plot_clustering(X, model, feature_names=None):
    """Visualise les résultats du clustering."""
    n_features = X.shape[1]
    
    # PCA si nécessaire
    if n_features > 2:
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
        explained_var = pca.explained_variance_ratio_
        xlabel, ylabel = f'PC1 ({explained_var[0]*100:.1f}%)', f'PC2 ({explained_var[1]*100:.1f}%)'
        
        centers_2d = pca.transform(model.cluster_centers)
        if model.representation_type == 'etalons':
            repr_2d = [pca.transform(etalons) for etalons in model.etalons]
        elif model.representation_type == 'axis' and model.cluster_axes:
            axes_2d = [{'center': pca.transform([ax['center']])[0], 
                        'direction': pca.components_ @ ax['direction']} for ax in model.cluster_axes]
        elif model.representation_type == 'distribution' and model.cluster_distributions:
            distributions_2d = []
            for d in model.cluster_distributions:
                if d['type'] == 'gaussian':
                    mean_2d = pca.transform([d['mean']])[0]
                    cov_2d = pca.components_ @ d['cov'] @ pca.components_.T
                    distributions_2d.append({'mean': mean_2d, 'cov': cov_2d, 'type': 'gaussian'})
                else:  # uniform
                    min_2d = pca.transform([d['min']])[0]
                    max_2d = pca.transform([d['max']])[0]
                    distributions_2d.append({'min': min_2d, 'max': max_2d, 'type': 'uniform'})
    else:
        X_2d = X
        xlabel = feature_names[0] if feature_names else 'Feature 1'
        ylabel = feature_names[1] if feature_names else 'Feature 2'
        centers_2d = model.cluster_centers
        if model.representation_type == 'etalons':
            repr_2d = model.etalons
        elif model.representation_type == 'axis' and model.cluster_axes:
            axes_2d = model.cluster_axes
        elif model.representation_type == 'distribution' and model.cluster_distributions:
            distributions_2d = model.cluster_distributions
    
    # Création du plot
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, model.k))
    
    # Points de données
    for i in range(model.k):
        mask = model.labels == i
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=[colors[i]], alpha=0.6, s=60, 
                  edgecolors='white', linewidth=0.5, label=f'Cluster {i+1}')
    
    # Représentations
    if model.representation_type == 'etalons':
        for i, etalons in enumerate(repr_2d):
            ax.scatter(etalons[:, 0], etalons[:, 1], c=[colors[i]], marker='s', s=200, 
                      edgecolors='black', linewidths=2.5, zorder=5)
            if len(etalons) > 1:
                sorted_idx = np.argsort(etalons[:, 0])
                ax.plot(etalons[sorted_idx, 0], etalons[sorted_idx, 1], 'k--', alpha=0.4, linewidth=1.5)
    
    elif model.representation_type == 'singleton':
        ax.scatter(centers_2d[:, 0], centers_2d[:, 1], c='red', marker='X', s=300, 
                  edgecolors='black', linewidths=2.5, label='Centroids', zorder=5)
    
    elif model.representation_type == 'axis' and 'axes_2d' in locals():
        for i, axis in enumerate(axes_2d):
            direction = axis['direction'] / np.linalg.norm(axis['direction'])
            scale = 2.0
            ax.plot([axis['center'][0] - scale*direction[0], axis['center'][0] + scale*direction[0]],
                   [axis['center'][1] - scale*direction[1], axis['center'][1] + scale*direction[1]],
                   color=colors[i], linewidth=3, alpha=0.8, zorder=4)
            ax.scatter(axis['center'][0], axis['center'][1], c='red', marker='o', s=150,
                      edgecolors='black', linewidths=2.5, zorder=5)
    
    elif model.representation_type == 'distribution' and 'distributions_2d' in locals():
        for i, dist in enumerate(distributions_2d):
            if dist['type'] == 'gaussian':
                if 'cov' in dist:
                    eigenvalues, eigenvectors = np.linalg.eig(dist['cov'])
                    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
                    width, height = 2 * np.sqrt(eigenvalues[0]), 2 * np.sqrt(eigenvalues[1])
                    ellipse = Ellipse(xy=dist['mean'], width=width, height=height, angle=angle, 
                                    alpha=0.2, color=colors[i])
                    ax.add_patch(ellipse)
                    ax.scatter(dist['mean'][0], dist['mean'][1], c='red', marker='D', s=100,
                              edgecolors='black', linewidths=2, zorder=5, label='Gaussian mean' if i == 0 else "")
                else:
                    ax.scatter(dist['mean'][0], dist['mean'][1], c='red', marker='D', s=100,
                              edgecolors='black', linewidths=2, zorder=5)
            
            else:  # distribution uniforme
                if 'min' in dist and 'max' in dist:
                    min_x, min_y = dist['min'][0], dist['min'][1]
                    max_x, max_y = dist['max'][0], dist['max'][1]
                    width = max_x - min_x
                    height = max_y - min_y
                    
                    rect = Rectangle((min_x, min_y), width, height,
                                   alpha=0.2, color=colors[i], linewidth=2, linestyle='--')
                    ax.add_patch(rect)
                    
                    center_x = (min_x + max_x) / 2
                    center_y = (min_y + max_y) / 2
                    ax.scatter(center_x, center_y, c='red', marker='s', s=100,
                              edgecolors='black', linewidths=2, zorder=5, label='Uniform center' if i == 0 else "")
    
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(f"Clustering - {model.representation_type.capitalize()} Representation", 
                fontsize=13, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig


# ============ APPLICATION STREAMLIT ============
def main():
    st.set_page_config(page_title="Nuées Dynamiques", page_icon="🔬", layout="wide")
    
    # Initialisation du session state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'clustering_done' not in st.session_state:
        st.session_state.clustering_done = False
    if 'quick_action' not in st.session_state:
        st.session_state.quick_action = None
    if 'force_upload' not in st.session_state:
        st.session_state.force_upload = False
    if 'force_sample' not in st.session_state:
        st.session_state.force_sample = False
    if 'X_for_model' not in st.session_state:
        st.session_state.X_for_model = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    
    st.title("🔬 Nuées Dynamiques - Multiple Representations")
    st.markdown("""
    Implémentation de l'algorithme de **Nuées Dynamiques** (Diday, 1971) avec différentes représentations de clusters.
    
    **Principe fondamental**: La distance d'un point à un ensemble (noyau) est définie comme la distance du point 
    au centre de gravité de cet ensemble: **d(x, E_i) = d(x, G_i)**
    """)
    
    # ============ SIDEBAR ============
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Vérifier si une action rapide a été déclenchée
        if st.session_state.force_upload:
            default_source = "Upload CSV"
            st.session_state.force_upload = False
        elif st.session_state.force_sample:
            default_source = "Generate Sample"
            st.session_state.force_sample = False
        else:
            default_source = "Upload CSV"
        
        # Source de données
        data_source_option = st.radio("📂 Data Source", ["Upload CSV", "Generate Sample"], 
                                     index=0 if default_source == "Upload CSV" else 1,
                                     key="data_source_option")
        
        X, feature_names, original_df = None, None, None
        
        # Section Upload CSV
        if data_source_option == "Upload CSV":
            uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'], key="csv_uploader")
            
            if uploaded_file is not None:
                try:
                    with st.spinner("Loading data..."):
                        df = pd.read_csv(uploaded_file)
                        original_df = df.copy()
                        
                        # Afficher info basique
                        st.success(f"✅ Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
                        
                        # Préprocesser les données
                        X, feature_names, prep_info = preprocess_data(df)
                        
                        if X is not None:
                            st.session_state.data_loaded = True
                            st.session_state.X = X
                            st.session_state.feature_names = feature_names
                            st.session_state.original_df = original_df
                            
                            # Réinitialiser le clustering si nouvelles données
                            if st.session_state.clustering_done and st.session_state.X_for_model is not None:
                                if X.shape[0] != st.session_state.X_for_model.shape[0]:
                                    st.session_state.clustering_done = False
                                    st.session_state.model = None
                                    st.session_state.X_for_model = None
                                    st.info("⚠️ New data loaded. Please run clustering again.")
                            
                            # Afficher info préprocessing
                            with st.expander("Preprocessing details"):
                                for info in prep_info:
                                    st.info(info)
                        else:
                            st.error("❌ Data preprocessing failed")
                            
                except Exception as e:
                    st.error(f"❌ Error loading file: {str(e)}")
        
        # Section Generate Sample
        else:
            dataset_type = st.selectbox("Dataset Type", 
                ["Blobs", "Elongated", "Moons", "Circles", "High-dimensional"],
                key="dataset_type")
            
            n_samples = st.slider("Number of samples", 100, 1000, 300, 50, key="n_samples")
            
            if st.button("Generate Data", type="primary", key="generate_btn"):
                with st.spinner("Generating data..."):
                    X, feature_names = generate_sample_data(dataset_type, n_samples)
                    st.session_state.data_loaded = True
                    st.session_state.X = X
                    st.session_state.feature_names = feature_names
                    st.session_state.original_df = None
                    
                    # Réinitialiser le clustering si nouvelles données
                    if st.session_state.clustering_done and st.session_state.X_for_model is not None:
                        if X.shape[0] != st.session_state.X_for_model.shape[0]:
                            st.session_state.clustering_done = False
                            st.session_state.model = None
                            st.session_state.X_for_model = None
                    
                    st.success(f"✅ Generated {X.shape[0]} samples with {X.shape[1]} features")
                    st.rerun()
        
        # Charger les données du session state
        if st.session_state.data_loaded:
            X = st.session_state.get('X')
            feature_names = st.session_state.get('feature_names')
            original_df = st.session_state.get('original_df')
        
        # Paramètres de clustering (seulement si données chargées)
        if st.session_state.data_loaded:
            st.markdown("---")
            st.header("🎯 Algorithm Parameters")
            
            representation_type = st.selectbox("Representation Type",
                ["etalons", "singleton", "axis", "distribution"],
                format_func=lambda x: {
                    "etalons": "Étalons (Multiple)", 
                    "singleton": "Singleton (K-means)", 
                    "axis": "Principal Axis", 
                    "distribution": "Distribution"
                }[x],
                key="repr_type")
            
            # Paramètres conditionnels
            if representation_type == 'etalons':
                n_etalons = st.slider("Étalons per cluster", 2, 20, 5, key="n_etalons")
                aggregation_type = st.selectbox("Aggregation function", ["simple", "diday"], 
                                               format_func=lambda x: "Simple" if x == "simple" else "Diday (original)",
                                               key="agg_type")
            elif representation_type == 'distribution':
                distribution_type = st.selectbox("Distribution type", ["gaussian", "uniform"],
                                                format_func=lambda x: x.capitalize(),
                                                key="dist_type")
                aggregation_type = "distribution"
                n_etalons = 5
            else:
                aggregation_type = "simple"
                n_etalons = 1
            
            k = st.slider("Number of clusters (K)", 2, 10, 3, key="k_clusters")
            max_iterations = st.slider("Max iterations", 10, 200, 100, 10, key="max_iter")
            random_state = st.number_input("Random seed", 0, 1000, 42, key="random_seed")
            
            # Bouton clustering
            if st.button("🚀 Run Clustering", type="primary", use_container_width=True, key="run_clustering"):
                if X is not None:
                    with st.spinner("Running Nuées Dynamiques algorithm..."):
                        # Préparer les paramètres du modèle
                        model_params = {
                            'k': k,
                            'n_etalons_per_cluster': n_etalons,
                            'max_iterations': max_iterations,
                            'tolerance': 1e-4,
                            'random_state': random_state,
                            'representation_type': representation_type,
                            'aggregation_type': aggregation_type
                        }
                        
                        if representation_type == 'distribution':
                            model_params['distribution_type'] = distribution_type
                        
                        # Créer et entraîner le modèle
                        model = NueesDynamiques(**model_params)
                        model.fit(X)
                        
                        # Sauvegarder les résultats
                        st.session_state.model = model
                        st.session_state.X_for_model = X.copy()
                        st.session_state.clustering_done = True
                        st.session_state.model_params = model_params
                        st.success("✅ Clustering completed!")
                        st.rerun()
                else:
                    st.error("❌ No data available. Please load or generate data first.")
    
    # ============ MAIN AREA ============
    # Vérifier si les boutons d'action rapide ont été cliqués
    if st.session_state.get('quick_action') == 'upload':
        st.session_state.force_upload = True
        st.session_state.quick_action = None
        st.rerun()
    elif st.session_state.get('quick_action') == 'sample':
        st.session_state.force_sample = True
        st.session_state.quick_action = None
        st.rerun()
    
    if st.session_state.data_loaded:
        X = st.session_state.get('X')
        feature_names = st.session_state.get('feature_names')
        original_df = st.session_state.get('original_df')
        
        # Onglets principaux
        tab1, tab2 = st.tabs(["📊 Data Overview", "📈 Clustering Results"])
        
        with tab1:
            st.subheader("Data Overview")
            
            # Afficher les données
            if original_df is not None:
                st.write("**Original Data Preview:**")
                st.dataframe(original_df.head(10), use_container_width=True)
            
            # Visualisation des données
            st.subheader("Data Visualization")
            
            if X is not None:
                if X.shape[1] == 2:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.scatter(X[:, 0], X[:, 1], alpha=0.6, s=50, c='steelblue', 
                              edgecolors='white', linewidth=0.5)
                    ax.set_xlabel(feature_names[0] if feature_names else 'Feature 1')
                    ax.set_ylabel(feature_names[1] if feature_names else 'Feature 2')
                    ax.set_title('Data Distribution (Scaled)')
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    plt.close()
                elif X.shape[1] > 2:
                    pca = PCA(n_components=2)
                    X_2d = pca.fit_transform(X)
                    var = pca.explained_variance_ratio_
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.scatter(X_2d[:, 0], X_2d[:, 1], alpha=0.6, s=50, c='steelblue', 
                              edgecolors='white', linewidth=0.5)
                    ax.set_xlabel(f'PC1 ({var[0]*100:.1f}%)')
                    ax.set_ylabel(f'PC2 ({var[1]*100:.1f}%)')
                    ax.set_title(f'PCA Projection ({(var[0]+var[1])*100:.1f}% variance)')
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    plt.close()
            
            # Statistiques
            if original_df is not None and feature_names:
                with st.expander("📈 Statistics Summary"):
                    stats_df = original_df[feature_names].describe()
                    st.dataframe(stats_df, use_container_width=True)
        
        with tab2:
            # Vérifier si le clustering a été fait sur les données actuelles
            if st.session_state.get('clustering_done', False) and 'model' in st.session_state:
                model = st.session_state.model
                X_for_model = st.session_state.get('X_for_model')
                
                # Vérifier si les données correspondent
                if X_for_model is not None and X is not None:
                    if X_for_model.shape[0] == X.shape[0]:
                        pass
                    else:
                        st.warning("⚠️ The clustering was performed on different data. Please run clustering again.")
                        st.session_state.clustering_done = False
                        st.session_state.model = None
                        st.session_state.X_for_model = None
                        st.stop()
                else:
                    st.warning("⚠️ No saved data found for this model. Please run clustering again.")
                    st.session_state.clustering_done = False
                    st.session_state.model = None
                    st.session_state.X_for_model = None
                    st.stop()
                
                # Utiliser les données sauvegardées avec le modèle
                X_for_model = st.session_state.get('X_for_model', X)
                
                # Métriques
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    if model.silhouette_score is not None:
                        st.metric("Silhouette Score", f"{model.silhouette_score:.4f}")
                    else:
                        st.metric("Silhouette Score", "N/A")
                with col2:
                    st.metric("Execution Time", f"{model.execution_time:.2f} ms")
                with col3:
                    try:
                        inertia = model.calculate_inertia(X_for_model)
                        st.metric("Inertia", f"{inertia:.2f}")
                    except Exception as e:
                        st.metric("Inertia", "N/A")
                with col4:
                    if model.davies_bouldin_score is not None:
                        st.metric("Davies-Bouldin Index", f"{model.davies_bouldin_score:.4f}")
                    else:
                        st.metric("Davies-Bouldin Index", "N/A")
                
                # Info sur la représentation
                st.subheader("🎯 Representation Details")
                rep_info = model.get_representation_info()
                if rep_info:
                    for info in rep_info:
                        st.info(info)
                else:
                    st.info("No representation information available.")
                
                # Visualisation du clustering
                st.subheader("🎨 Clustering Visualization")
                try:
                    fig = plot_clustering(X_for_model, model, feature_names)
                    st.pyplot(fig)
                    plt.close()
                except Exception as e:
                    st.error(f"Error creating visualization: {str(e)}")
                    # Fallback: simple scatter plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    for i in range(model.k):
                        mask = model.labels == i
                        if X_for_model.shape[1] >= 2:
                            ax.scatter(X_for_model[mask, 0], X_for_model[mask, 1], 
                                      label=f'Cluster {i+1}')
                        else:
                            ax.scatter(np.arange(np.sum(mask)), X_for_model[mask, 0], 
                                      label=f'Cluster {i+1}')
                    ax.set_title("Simple Cluster Visualization")
                    ax.legend()
                    st.pyplot(fig)
                    plt.close()
                
                # Légende
                legends = {
                    'etalons': "**Legend**: Squares = étalons, Dashed lines = skeleton",
                    'singleton': "**Legend**: Red X = centroids (centres de gravité)",
                    'axis': "**Legend**: Lines = principal axes, Red circles = axis centers",
                    'distribution': "**Legend**: Ellipses/rectangles = confidence regions, Red markers = distribution centers"
                }
                st.info(legends.get(model.representation_type, ""))
                
                # Statistiques des clusters
                st.subheader("📊 Cluster Statistics")
                cluster_stats = []
                for i in range(model.k):
                    size = np.sum(model.labels == i)
                    percentage = 100 * size / len(X_for_model) if len(X_for_model) > 0 else 0
                    
                    stats = {
                        'Cluster': f"C{i+1}",
                        'Size': size,
                        'Percentage': f"{percentage:.1f}%"
                    }
                    
                    cluster_stats.append(stats)
                
                stats_df = pd.DataFrame(cluster_stats)
                st.dataframe(stats_df, use_container_width=True)
                
                # Résultats détaillés
                st.subheader("💾 Results Download")
                
                # Préparer les données pour export
                if original_df is not None:
                    results_df = original_df.copy()
                else:
                    if feature_names and len(feature_names) == X_for_model.shape[1]:
                        results_df = pd.DataFrame(X_for_model, columns=feature_names)
                    else:
                        results_df = pd.DataFrame(X_for_model, columns=[f'Feature_{i+1}' for i in range(X_for_model.shape[1])])
                
                results_df['Cluster'] = model.labels + 1
                results_df['Cluster_Label'] = ['C' + str(label) for label in results_df['Cluster']]
                
                # Aperçu
                with st.expander("👁️ Preview Results"):
                    st.dataframe(results_df.head(15), use_container_width=True)
                
                # Bouton de téléchargement
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"nuees_dynamiques_{model.representation_type}_results.csv",
                    mime="text/csv",
                    type="primary"
                )
                
                # Bouton de réinitialisation
                if st.button("🔄 Reset Clustering", type="secondary"):
                    st.session_state.clustering_done = False
                    st.session_state.model = None
                    st.session_state.X_for_model = None
                    st.rerun()
            
            else:
                st.info("👈 Configure clustering parameters in the sidebar and click 'Run Clustering' to see results.")
    
    else:
        # Page d'accueil
        st.markdown("""
        ## Welcome to Nuées Dynamiques Explorer
        
        **Nuées Dynamiques** (Dynamic Clouds) is an advanced clustering algorithm 
        introduced by Édouard Diday in 1971 that uses multiple representative points 
        per cluster instead of a single centroid.
        
        ### Key Principle:
        
        The distance from a point to a cluster is defined as the distance from the point 
        to the **center of gravity (centroid)** of that cluster:
        
        **d(x, E_i) = d(x, G_i)**
        
        where G_i is the center of gravity of cluster i.
        
        ### Getting Started:
        
        1. **Choose your data source** in the sidebar:
           - 📁 **Upload CSV**: Upload your own dataset
           - 🔧 **Generate Sample**: Use predefined datasets
        
        2. **Configure algorithm parameters**:
           - Select representation type (Étalons, Singleton, Axis, Distribution)
           - Set number of clusters and other parameters
           - Click "Run Clustering"
        
        3. **Explore results**:
           - View clustering visualization
           - Analyze cluster statistics
           - Download results
        
        ### Available Representations:
        
        - **Étalons (original)**: Multiple representative points per cluster
        - **Singleton**: Single centroid (equivalent to K-means)
        - **Axis**: Principal direction of each cluster
        - **Distribution**: Probability distribution model
        
        ### Sample Datasets:
        
        - **Blobs**: Spherical clusters (good for K-means)
        - **Elongated**: Linear clusters (showcases étalons advantage)
        - **Moons**: Non-linear separated clusters
        - **Circles**: Concentric circles
        - **High-dimensional**: 10D data with PCA visualization
        
        **Reference:** Diday, E. (1971). *Une nouvelle méthode en classification automatique 
        et reconnaissance des formes: la méthode des nuées dynamiques*. Revue de Statistique Appliquée, 19(2), 19-33.
        """)
        
        # Quick actions
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📁 Start with CSV Upload", use_container_width=True, key="quick_upload"):
                st.session_state.quick_action = 'upload'
                st.rerun()
        
        with col2:
            if st.button("🔧 Start with Sample Data", use_container_width=True, key="quick_sample"):
                st.session_state.quick_action = 'sample'
                st.rerun()

# ============ EXECUTION ============
if __name__ == "__main__":
    main()
