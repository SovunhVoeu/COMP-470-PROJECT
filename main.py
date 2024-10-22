!pip install scikit-learn
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
# Load the California housing dataset
california_housing = fetch_california_housing()
# Create a dataframe to hold data and add Longitude and Latitude
data = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
X, y = make_blobs(random_state = 42)
kmeans = KMeans(n_clusters=5, random_state=42)
silhouette_score(X, kmeans.fit_predict(x))
