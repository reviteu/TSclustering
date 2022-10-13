#Import Modules
from tokenize import group
from unicodedata import name
import pandas as pd
import numpy as np
from dtaidistance import dtw
from tslearn.clustering import TimeSeriesKMeans, silhouette_score, KernelKMeans
from tslearn import metrics
from tsfresh import extract_features
from scipy.cluster.hierarchy import single, complete, average, ward, dendrogram
from scipy.cluster.hierarchy import fcluster
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
from functools import partial, reduce


class ClusterWithFeatures():

    def __init__(self, data_path, cluster_column, item_column):
        self.data_path = data_path
        self.cluster_column = cluster_column
        self.item_column = item_column

    def load_data(self):
        """Load CSV from path

        Returns:
            data : dataframe with weekly seasonal index for different items
        """

        df = pd.read_excel(self.data_path)
        df = df.drop('Total', axis=1)
        data = df.copy(deep=True)
        data['cluster'] = data.iloc[:,1:].apply(lambda s: s.to_numpy(), axis=1)
        cluster_df = data[['Item', 'cluster']]


        return df, cluster_df

    def extract_ts_features(self):
        """Extracts all the feature set for the timeseries input with the tsfresh library. 

        Returns:
            Dataframe: Cleaned dataframe with removing inf and NaNs and appended feature columns
        """

        data, cluster_data = self.load_data()
        melted_data = data.melt(id_vars="Item", var_name="week", value_name="cluster")
        melted_data = melted_data.sort_values(["Item", "week"]).reset_index(drop=True)
        extracted_features = extract_features(melted_data, column_id=self.item_column, column_value=self.cluster_column)
        extracted_features = extracted_features.reset_index()
        extracted_features = extracted_features.drop(['index'], axis=1)

        print('Features Extracted...... Cleaning Features')

        cleaned_features = extracted_features.mask(np.isinf(extracted_features))
        cleaned_features = cleaned_features.interpolate(method = 'linear').ffill().bfill()
        cleaned_features = cleaned_features.dropna(axis=1)

        print('Features Cleaned')

        return cleaned_features

    def reduce_dimensions(self):
        """Dimensionality reduction of the extracted feature data with PCA

        Returns:
            data array: decomposed pca features generated from the generated feature set with scipy
        """

        data = self.extract_ts_features()

        min_max_scaler = preprocessing.StandardScaler()
        np_scaled = min_max_scaler.fit_transform(data)
        x_normalized = pd.DataFrame(np_scaled, columns = data.columns)
        pca = PCA(n_components=0.99)

        pca.fit(x_normalized)
        reduced_data = pca.transform(x_normalized)
        print(f'Data reduced to size: {reduced_data.shape} from size: {data.shape}')

        return reduced_data

    
    def hierarchical_clustering(self, dist_mat, method='ward'):
        """Implementation of hierarchical clustering model

        Args:
            dist_mat (array): matrix containing timeseries, distance or decomposed features
            method (str, optional): linkage method for hierarchical_clustering. Defaults to 'ward'.

        Returns:
            model: Cluster model that can be used to get labels for a given data
        """

        print('Starting hierarchical clustering ')
        if method == 'complete':
            Z = complete(dist_mat)
        if method == 'single':
            Z = single(dist_mat)
        if method == 'average':
            Z = average(dist_mat)
        if method == 'ward':
            Z = ward(dist_mat)
        
        fig = plt.figure(figsize=(30, 20))
        dn = dendrogram(Z, truncate_mode='level')
        plt.title(f"Dendrogram for {method}-linkage with correlation distance")
        plt.savefig('dendrogram.png')

       
        return Z



    def kmeans_clustering(self, dist_mat):
        """Implementation of kmeans clustering model

        Args:
            dist_mat (array): matrix containing timeseries, distance or decomposed features

        Returns:
            model: Cluster model that can be used to get labels for a given data
        """

        print('Starting kmeans clustering ')
        kmeans = KMeans(n_clusters=14, random_state=42)
        kmeans.fit_predict(dist_mat)
        labels = kmeans.labels_


        df = pd.DataFrame({'x':labels}) 

        return df
    
    def plot_clusters(self, data):
        
        sns.set(style='darkgrid')
        ax = sns.countplot(x="Item", data=data)
        plt.show()
        
        return ax

class ClusterWithTS(ClusterWithFeatures):

    def __init__(self, data_path, cluster_column, item_column):
        self.data_path = data_path
        self.cluster_column = cluster_column
        self.item_column = item_column

    def dtw_transform(self):
        """Return distance matrix computed using dynamic time warping for a given input timeseries as a list of arrays

        Returns:
            array: distance matrix of size (N * M) where N is the number of time series and M is the common lenght of the time series
        """

        data, grouped_data = self.load_data()

        timeseries = list(grouped_data[self.cluster_column].iloc[:])
        d = dtw.distance_matrix_fast(timeseries)

        return d

    
    def kmeans_ts_clustering(self):
        """Train Kmeans clustering model

        Returns:
            model object: Kmeans models to predict lables on a given data
        """

        d = self.dtw_transform()
        km = TimeSeriesKMeans(n_clusters=14, verbose=False, random_state=55)
        y_pred_km = km.fit_predict(d)

        return y_pred_km

    def kernelmeans_ts_clustering(self):
        """Train Kernel means clustering model

        Returns:
            model object: Kernel means models to predict lables on a given data
        """

        d = self.dtw_transform()
        kt = KernelKMeans(n_clusters=14, verbose=False, random_state=55, n_jobs=-1)
        y_pred_kt = kt.fit_predict(d)

        return y_pred_kt


def main(data_type, train_type):
    
    if data_type == "overall":
        data_path = "overall_index.xlsx"
    elif data_type == "launch":
        data_path = "launch_index.xlsx"
    elif data_type == "rol":
        data_path = "rol_index.xlsx"
        
        
    cluster_feature = ClusterWithFeatures(data_path, 'cluster', 'Item' )    
    cluster_TS = ClusterWithTS(data_path, 'cluster', 'Item')
    
    if train_type == 'ts_clustering':
        

        kmeans_clustering_label = cluster_TS.kmeans_ts_clustering()
        kernelmeans_clustering_label = cluster_TS.kernelmeans_ts_clustering()

        data, grouped_data = cluster_TS.load_data()

        grouped_data[f'kmeans_{data_type}'] = kmeans_clustering_label
        grouped_data[f'hierarchical_clustering{data_type}'] = kernelmeans_clustering_label

        final_cluster_data = grouped_data.drop('cluster', axis=1)


    elif train_type == 'feature_clustering':

        data_matrix = cluster_feature.reduce_dimensions()

        linkage_matrix = cluster_feature.hierarchical_clustering(data_matrix)

        hierarchical_clustering_label = fcluster(linkage_matrix, 80 , criterion='distance')

        kmeans_clustering_label = cluster_feature.kmeans_clustering(data_matrix)

        data, grouped_data = cluster_feature.load_data()

        grouped_data[f'kmeans_{data_type}'] = kmeans_clustering_label
        grouped_data[f'hierarchical_clustering_{data_type}'] = hierarchical_clustering_label
        
        final_cluster_data = grouped_data.drop('cluster', axis=1)
        final_cluster_data[f'kmeans_{data_type}'] = "cluster" + " "+  final_cluster_data[f'kmeans_{data_type}'].astype(str)
        final_cluster_data[f'hierarchical_clustering_{data_type}'] = "cluster" + " " + final_cluster_data[f'hierarchical_clustering_{data_type}'].astype(str)

        
    return final_cluster_data

if __name__ == "__main__":

    cluster_data_overall = main("overall", "feature_clustering")
    cluster_data_launch = main("launch", "feature_clustering")
    cluster_data_rol = main("rol", "feature_clustering")
    
    combined_data = [cluster_data_overall, cluster_data_launch, cluster_data_rol]
    merge = partial(pd.merge, on = "Item", how="outer")
    cluster_data = reduce(merge, combined_data)
    
    cluster_data.to_csv('clusters.csv')
