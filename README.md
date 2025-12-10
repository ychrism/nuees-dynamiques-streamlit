# 📊 Nuées Dynamiques - Clustering Explorer

**Nuées Dynamiques** (Dynamic Clouds) is an interactive web application implementing Édouard Diday's 1971 clustering algorithm with multiple cluster representations. This tool provides an intuitive interface for exploring different clustering approaches and visualizing results in real-time.

## 🌟 Features

### 🔬 Multiple Cluster Representations
- **Étalons (Original)**: Multiple representative points per cluster
- **Singleton (K-means)**: Single centroid per cluster
- **Principal Axis**: Cluster represented by main direction
- **Probability Distribution**: Gaussian or uniform distribution models

### 📊 Data Handling
- **CSV Upload**: Upload your own datasets
- **Sample Datasets**: Built-in datasets for testing
- **Automatic Preprocessing**: Missing value imputation and feature scaling
- **PCA Visualization**: Automatic dimensionality reduction for high-dimensional data

### 🎨 Interactive Visualization
- Real-time clustering visualization
- Multiple representation-specific visualizations
- Cluster statistics and metrics
- Comparative analysis tools

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/ychrism/nuees-dynamiques-streamlit.git
cd nuees-dynamiques-streamlit
```

2. **Create a virtual environment:**
```bash
python -m venv nuees_env
```

3. **Activate the environment:**
- **Windows:**
```bash
nuees_env\Scripts\activate
```
- **macOS/Linux:**
```bash
source nuees_env/bin/activate
```

4. **Install dependencies:**
```bash
pip install -r requirements.txt
```


### Running the Application

```bash
streamlit run app.py
```

The application will open automatically in your default web browser at `http://localhost:8501`.

## 📖 User Guide

### Step 1: Choose Data Source
- **Upload CSV**: Upload your dataset in CSV format
- **Generate Sample**: Use predefined datasets:
  - **Blobs**: Spherical clusters (ideal for K-means)
  - **Elongated**: Linear clusters (showcases étalons advantage)
  - **Moons**: Non-linear separated clusters
  - **Circles**: Concentric circles
  - **High-dimensional**: 10D data with PCA visualization

### Step 2: Configure Algorithm Parameters
- **Number of Clusters (K)**: 2 to 10 clusters
- **Representation Type**: Choose from 4 representations
- **Étalons per Cluster**: 1 to 20 representative points
- **Aggregation Function**: Simple or Diday's original function
- **Distribution Type**: Gaussian or Uniform (for distribution representation)

### Step 3: Run Clustering
Click the **"🚀 Run Clustering"** button to execute the algorithm. The application will display:
- Clustering visualization
- Inertia (within-cluster sum of squares)
- Iteration count
- Cluster statistics
- Representation details

### Step 4: Export Results
Download clustering results as CSV for further analysis.

## 🧮 Algorithm Details

### Nuées Dynamiques (Diday, 1971)
The algorithm uses multiple representative points (étalons) per cluster instead of a single centroid. Key features:

1. **Initialization**: Random selection of étalons
2. **Assignment**: Points assigned to nearest étalon
3. **Update**: Étalons updated based on aggregation function
4. **Convergence**: Iterate until stable

### Mathematical Formulation
- Distance function: $D(x, E_i) = \min_{y \in E_i} \|x - y\|$
- Aggregation function: $R(x, i, L) = \frac{D(x, E_i) \cdot D(x, C_i)}{[\sum_{j=1}^K D(x, E_j)]^2}$
- Update: $E_i^{(n+1)} = \arg\min_{n_i \text{ points}} R(x, i, L^{(n)})$

## 🎯 Use Cases

### Research & Education
- Teaching clustering algorithms
- Comparing different representation methods
- Visualizing high-dimensional data

### Data Analysis
- Exploratory data analysis
- Pattern recognition
- Customer segmentation
- Anomaly detection

### Algorithm Development
- Testing new clustering approaches
- Parameter optimization
- Performance comparison

## 📊 Sample Datasets

### 1. Blobs Dataset
- **Purpose**: Test spherical clustering
- **Best for**: Singleton (K-means) representation
- **Features**: 2D, 3 clusters, Gaussian distribution

### 2. Elongated Dataset
- **Purpose**: Test linear clustering
- **Best for**: Étalons or Axis representation
- **Features**: 2D, elongated clusters

### 3. Moons Dataset
- **Purpose**: Test non-linear separation
- **Best for**: Étalons representation
- **Features**: 2D, crescent-shaped clusters

### 4. Circles Dataset
- **Purpose**: Test concentric clustering
- **Best for**: Étalons or Distribution representation
- **Features**: 2D, concentric circles

### 5. High-dimensional Dataset
- **Purpose**: Test dimensionality reduction
- **Best for**: All representations with PCA
- **Features**: 10D, 4 clusters

## 🔧 Technical Details

### Dependencies
- **Streamlit**: Web application framework
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation
- **Matplotlib**: Visualization
- **Scikit-learn**: Machine learning utilities
- **SciPy**: Scientific computing


### Performance
- **Memory**: ~5N where N is number of samples
- **Speed**: Efficient implementation with numpy
- **Scalability**: Handles thousands of samples

## 📚 References

1. **Diday, E. (1971)**: *Une nouvelle méthode en classification automatique et reconnaissance des formes: la méthode des nuées dynamiques*. Revue de Statistique Appliquée, 19(2), 19-33.

2. **Related Methods**:
   - K-means clustering
   - Gaussian Mixture Models
   - Principal Component Analysis
   - Model-based clustering


## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📧 Contact

For questions or support:
- **GitHub Issues**: [Report bugs or request features](https://github.com/ychrism/nuees-dynamiques-streamlit/issues)
- **Email**: yvescmedagbe@gmail.com

## 🙏 Acknowledgments

- Édouard Diday for the original algorithm
- The Streamlit team for the amazing framework
- Contributors and testers

---

**Happy Clustering!** 🎯

*Note: This tool is for educational and research purposes. Always validate clustering results with domain knowledge.*