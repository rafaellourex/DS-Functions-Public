import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

sns.set_style("whitegrid")


def get_ss_variables(df):
    """Get the SS for each variable"""
    ss_vars = df.var() * (df.count() - 1)
    return ss_vars


def r2_variables(df, labels):
    """Get the R² for each variable"""
    sst_vars = get_ss_variables(df)
    ssw_vars = np.sum(df.groupby(labels).apply(get_ss_variables))
    return 1 - ssw_vars / sst_vars


def metric_evaluation(df, label_name):
    from sklearn.metrics import davies_bouldin_score  # the lower the better

    db = davies_bouldin_score(df.drop(columns=[f"{label_name}"]), df[label_name])
    from sklearn.metrics import silhouette_score  # the higher the better (closer to 1)

    ss = silhouette_score(df.drop(columns=[f"{label_name}"]), df[label_name])
    from sklearn.metrics import calinski_harabasz_score  # the higher the better

    ch = calinski_harabasz_score(df.drop(columns=[f"{label_name}"]), df[label_name])

    return print(
        "Davies-bouldin score :",
        np.round(db, 4),
        "\nSilhoutte score :",
        np.round(ss, 4),
        "\nCalinski-Harabasz score : ",
        np.round(ch, 4),
    )


def get_ss(df):
    ss = np.sum(df.var() * (df.shape[0] - 1))
    return ss  # return sum of sum of squares of each df variable


def multiple_radar_plot(data, columns):
    labels = list(columns)
    labels = [*labels, labels[0]]
    col = data.iloc[1, :]
    col = [*col, col[0]]
    label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(columns) + 1)

    plt.figure(figsize=(8, 8))
    plt.subplot(polar=True)
    for i in range(len(data)):

        df = data.iloc[i, :]
        df = [*df, df[0]]
        plt.plot(label_loc, df, label=f"Cluster_{i+1}")

    lines, labels = plt.thetagrids(np.degrees(label_loc), labels=labels)
    plt.xticks(rotation=90)
    plt.legend()
    plt.show()


def clustering_profile(data, labels):
    data1 = data.drop(f"{labels}", axis=1).copy()

    print("Boxplot - Comparison")
    box1 = data.copy()
    box1["Cons"] = 1
    for i in data1.columns:
        plt.figure(figsize=(10, 7))
        sns.boxplot(data=box1, x=box1["Cons"], y=box1[i], hue=box1[f"{labels}"])
        plt.legend(bbox_to_anchor=(1.05, 1), borderaxespad=0.0)
        plt.title(f"{labels} - {i} Distribution")
        plt.tight_layout()
        # plt.savefig(f'{labels} - {i} Boxplot Distribution.png',dpi=150)
        plt.show()

    print("Histograms - Comparison")
    for i in data1.columns:
        plt.figure(figsize=(12, 9))
        # fig, ax = plt.subplots(figsize = (12,9))

        sns.histplot(data=data, x=data[i], hue=data[f"{labels}"], palette="colorblind")
        plt.title(f"{labels} - {i}  Distribution")
        # plt.legend(bbox_to_anchor=(1.05, 1), borderaxespad=0.)
        plt.tight_layout()
        # plt.savefig(f'{labels} - {i} Histogram Distribution.png',dpi=150)
        plt.show()


def get_r2_hc(df, link_method, max_nclus, min_nclus=1, dist="euclidean"):
    def get_ss(df):
        ss = np.sum(df.var() * (df.count() - 1))
        return ss  # return sum of sum of squares of each df variable

    sst = get_ss(df)  # get total sum of squares

    r2 = []  # where we will store the R2 metrics for each cluster solution

    for i in range(min_nclus, max_nclus + 1):  # iterate over desired ncluster range
        cluster = AgglomerativeClustering(
            n_clusters=i, affinity=dist, linkage=link_method
        )

        hclabels = cluster.fit_predict(df)  # get cluster labels

        df_concat = pd.concat(
            (df, pd.Series(hclabels, name="labels")), axis=1
        )  # concat df with labels

        ssw_labels = df_concat.groupby(by="labels").apply(
            get_ss
        )  # compute ssw for each cluster labels

        ssb = sst - np.sum(ssw_labels)  # remember: SST = SSW + SSB

        r2.append(ssb / sst)  # save the R2 of the given cluster solution

    return np.array(r2)


def calculate_r2(df, labels, cluster_method):

    def get_ss(df):
        ss = np.sum(df.var() * (df.count() - 1))
        return ss  # return sum of sum of squares of each df variable

    sst = get_ss(df)  # get total sum of squares
    df_concat = pd.concat(
        [df, pd.Series(labels, index=df.index, name=f"{cluster_method}_labels")], axis=1
    )

    ssw_labels = df_concat.groupby(by=f"{cluster_method}_labels").apply(
        get_ss
    )  # compute ssw for each cluster labels
    ssb = sst - np.sum(ssw_labels)  # remember: SST = SSW + SSB
    r2 = ssb / sst
    print("Cluster solution with R^2 of %0.4f" % r2)


def return_r2(df, labels):
    def get_ss(df):
        ss = np.sum(df.var() * (df.count() - 1))
        return ss  # return sum of sum of squares of each df variable

    sst = get_ss(df)  # get total sum of squares
    df_concat = pd.concat(
        [df, pd.Series(labels, index=df.index, name=f"labels")], axis=1
    )

    ssw_labels = df_concat.groupby(by=f"labels").apply(
        get_ss
    )  # compute ssw for each cluster labels
    ssb = sst - np.sum(ssw_labels)  # remember: SST = SSW + SSB
    r2 = ssb / sst
    return r2


def multiple_R2_HCluster(data, k_range, linkage):

    results = []
    for i in k_range + 1:
        cluster_solution = AgglomerativeClustering(n_clusters=i, linkage=linkage)
        cluster_labels = cluster_solution.fit_predict(data)
        results.append(return_r2(data, cluster_labels))
        # print(calculate_r2(data,cluster_labels,'cluster_R2'))
    return results


# Hierarchical Clustering
class HierarchicalCluster_:
    def __init__(self, data, rangeMax=15, linkage="ward", threshold=50):
        self.data = data
        self.ranges = np.arange(1, rangeMax + 1)
        self.linkage = linkage
        self.threshold = threshold
        self.clusterModel = AgglomerativeClustering(
            n_clusters=None, linkage=linkage, distance_threshold=threshold
        )

    def hierarquicalLinkage_Method(self):
        data = self.data
        range_clusters = self.ranges

        result = pd.DataFrame()
        for i in ["ward", "complete", "average", "single"]:
            result[i] = multiple_R2_HCluster(data, range_clusters, i)

        fig = plt.figure(figsize=(11, 5))
        sns.lineplot(data=result, linewidth=2.5, markers=["o"] * 4)
        fig.suptitle("R2 plot for various hierarchical methods", fontsize=21)
        plt.gca().invert_xaxis()  # invert x axis
        plt.legend(title="HC methods", title_fontsize=11)
        plt.xticks(range_clusters)
        plt.xlabel("Number of clusters", fontsize=13)
        plt.ylabel("R2 metric", fontsize=13)

        plt.show()
        return result

    def dendogram(self, distance="Euclidean"):

        from scipy.cluster.hierarchy import dendrogram

        data = self.data
        hclust = self.clusterModel
        threshold = self.threshold
        linkage = self.linkage

        hclust.fit(data)

        # Adapted from:
        # https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html#sphx-glr-auto-examples-cluster-plot-agglomerative-dendrogram-py

        # create the counts of samples under each node (number of points being merged)
        counts = np.zeros(hclust.children_.shape[0])
        n_samples = len(hclust.labels_)

        # hclust.children_ contains the observation ids that are being merged together
        # At the i-th iteration, children[i][0] and children[i][1] are merged to form node n_samples + i
        for i, merge in enumerate(hclust.children_):
            # track the number of observations in the current cluster being formed
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    # If this is True, then we are merging an observation
                    current_count += 1  # leaf node
                else:
                    # Otherwise, we are merging a previously formed cluster
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        # the hclust.children_ is used to indicate the two points/clusters being merged (dendrogram's u-joins)
        # the hclust.distances_ indicates the distance between the two points/clusters (height of the u-joins)
        # the counts indicate the number of points being merged (dendrogram's x-axis)
        linkage_matrix = np.column_stack(
            [hclust.children_, hclust.distances_, counts]
        ).astype(float)

        # Plot the corresponding dendrogram

        fig = plt.figure(figsize=(20, 15))
        # The Dendrogram parameters need to be tuned
        y_threshold = threshold
        dendrogram(
            linkage_matrix,
            truncate_mode="level",
            p=5,
            color_threshold=y_threshold,
            above_threshold_color="k",
        )
        plt.hlines(y_threshold, 0, 1000, colors="r", linestyles="dashed")
        plt.title(f"Hierarchical Clustering - {linkage}'s Dendrogram", fontsize=21)
        plt.xlabel("Number of points in node (or index of point if no parenthesis)")
        plt.ylabel(f"{distance} Distance", fontsize=13)
        plt.show()


def hc_chooseClusters(data, maxRange, linkage="ward", threshold=50):
    """""" """""" """
    
    Receives:
        data - pandas df containing the data that will be used for clustering 
        maxRange - maximum number of clusters to investigate
    
    """ """""" """"""
    cluster_obj = HierarchicalCluster_(
        data, rangeMax=maxRange, linkage=linkage, threshold=threshold
    )

    print("Plot R2 of different linkage methods")
    print(" ")
    cluster_obj.hierarquicalLinkage_Method()
    print(" ")
    print("Plot dendogram")
    avg_silhouette = cluster_obj.dendogram()


class performHierarchicalCluster:

    def __init__(self, data, nClusters=5, linkage="ward", affinity="euclidean"):

        print("Performing Hierarchical Clustering Cluster Analysis")
        print(f"Number of clusters: {nClusters}")
        print(f"Linkage Method: {linkage}")
        print(f"Distance Metric: {affinity}")

        self.data = data
        self.nClusters = nClusters
        self.linkage = linkage
        self.affinity = affinity
        self.clusterModel = AgglomerativeClustering(
            n_clusters=nClusters, linkage=linkage, affinity=affinity
        )

    def fitHC(self):
        data = self.data
        clusterModel = self.clusterModel
        clusterModel.fit(data)

        return clusterModel

    def performPredictions(self):

        dataOrig = self.data
        dataAssess = dataOrig.copy()
        clusterModel = self.clusterModel

        clusterPredictions = clusterModel.fit_predict(dataOrig)
        dataAssess["KMeansLabels"] = clusterPredictions

        print("Number of observation per cluster")
        display(dataAssess.groupby("KMeansLabels").count().iloc[:, 0])

        print(" ")
        print("Cluster Means:")
        display(dataAssess.groupby("KMeansLabels").mean())

        print(" ")
        print("Cluster Standard Deviation:")
        display(dataAssess.groupby("KMeansLabels").std())

        # get total variance
        sst = get_ss(dataOrig)
        # compute ssw for each cluster labels
        ssw_labels = dataAssess.groupby(by="KMeansLabels").apply(get_ss)
        # Obtain SSB. Remember: SST = SSW + SSB
        ssb = sst - np.sum(ssw_labels)
        r2 = ssb / sst

        print(f"K-Means R2: {r2}")
        print(" ")
        features_r2 = r2_variables(dataAssess, "KMeansLabels")[:-1]

        print("Evaluating Cluster Performance:")

        metric_evaluation(dataAssess, "KMeansLabels")
        print(" ")

        print("Checking how much variance of each feature is explained by the cluster:")
        print(features_r2)

        clusterProfile = dataAssess.groupby("KMeansLabels").mean()
        multiple_radar_plot(clusterProfile, clusterProfile.columns)

        return (dataAssess, clusterModel)


# K-Means Clustering
class assessKMeans:
    def __init__(self, data, rangeMax=15):
        self.data = data
        self.ranges = np.arange(1, rangeMax + 1)

    def inertiaPlot(self):

        data = self.data
        range_clusters = self.ranges

        inertia = []
        for i in range_clusters:  # iterate over desired ncluster range
            kmclust = KMeans(n_clusters=i, init="k-means++", n_init=15, random_state=1)
            kmclust.fit(data)
            # CODE HERE
            inertia.append(
                kmclust.inertia_
            )  # save the inertia of the given cluster solution

        plt.figure(figsize=(15, 10))
        plt.plot(pd.Series(inertia, index=range_clusters))
        plt.ylabel("Inertia: SSw")
        plt.xlabel("Number of clusters")
        plt.title("Inertia plot over clusters", size=15)
        plt.show()

    def silhoetteScore(self):
        from os.path import join
        from sklearn.metrics import silhouette_samples, silhouette_score
        import matplotlib.cm as cm

        data = self.data
        range_clusters = self.ranges
        # Adapted from:
        # https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#sphx-glr-auto-examples-cluster-plot-kmeans-silhouette-analysis-py
        # Storing average silhouette metric
        avg_silhouette = []
        for nclus in range_clusters:
            # Skip nclus == 1
            if nclus == 1:
                continue

            # Create a figure
            fig = plt.figure(figsize=(13, 7))

            # Initialize the KMeans object with n_clusters value and a random generator
            # seed of 10 for reproducibility.
            kmclust = KMeans(
                n_clusters=nclus, init="k-means++", n_init=15, random_state=1
            )
            cluster_labels = kmclust.fit_predict(data)

            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed clusters
            silhouette_avg = silhouette_score(data, cluster_labels)
            avg_silhouette.append(silhouette_avg)
            print(
                f"For n_clusters = {nclus}, the average silhouette_score is : {silhouette_avg}"
            )

            # Compute the silhouette scores for each sample
            sample_silhouette_values = silhouette_samples(data, cluster_labels)

            y_lower = 10
            for i in range(nclus):
                # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
                ith_cluster_silhouette_values = sample_silhouette_values[
                    cluster_labels == i
                ]
                ith_cluster_silhouette_values.sort()

                # Get y_upper to demarcate silhouette y range size
                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                # Filling the silhouette
                color = cm.nipy_spectral(float(i) / nclus)
                plt.fill_betweenx(
                    np.arange(y_lower, y_upper),
                    0,
                    ith_cluster_silhouette_values,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.7,
                )

                # Label the silhouette plots with their cluster numbers at the middle
                plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            plt.title("The silhouette plot for the various clusters.")
            plt.xlabel("The silhouette coefficient values")
            plt.ylabel("Cluster label")

            # The vertical line for average silhouette score of all the values
            plt.axvline(x=silhouette_avg, color="red", linestyle="--")

            # The silhouette coefficient can range from -1, 1
            xmin, xmax = np.round(sample_silhouette_values.min() - 0.1, 2), np.round(
                sample_silhouette_values.max() + 0.1, 2
            )
            plt.xlim([xmin, xmax])

            # The (nclus+1)*10 is for inserting blank space between silhouette
            # plots of individual clusters, to demarcate them clearly.
            plt.ylim([0, len(data) + (nclus + 1) * 10])

            plt.yticks([])  # Clear the yaxis labels / ticks
            plt.xticks(np.arange(xmin, xmax, 0.1))
            plt.tight_layout()
            # plt.savefig('Silhouette_Score.png')
        return avg_silhouette

    def get_r2_kmeans(self):

        df = self.data
        range_clusters = self.ranges

        sst = get_ss(df)  # get total sum of squares

        r2 = []  # where we will store the R2 metrics for each cluster solution

        for i in range_clusters:  # iterate over desired ncluster range
            cluster = KMeans(n_clusters=i, init="k-means++", n_init=15, random_state=0)

            # get cluster labels
            hclabels = cluster.fit_predict(df)

            df_assess = df.copy()
            df_assess["labels"] = hclabels

            # compute ssw for each cluster labels
            ssw_labels = df_assess.groupby(by="labels").apply(get_ss)

            # Obtain SSB. Remember: SST = SSW + SSB
            ssb = sst - np.sum(ssw_labels)
            r2_ = ssb / sst

            # append the R2 of the given cluster solution
            r2.append(r2_)

        print(r2)
        return np.array(r2)


def kMeans_chooseClusters(data, maxRange):
    """""" """""" """
    
    Receives:
        data - pandas df containing the data that will be used for clustering 
        maxRange - maximum number of clusters to investigate
    
    """ """""" """"""
    cluster_obj = assessKMeans(data, maxRange)
    print("Intertia Plot")
    print(" ")
    cluster_obj.inertiaPlot()
    print(" ")
    print("Shiloette Score")
    avg_silhouette = cluster_obj.silhoetteScore()
    print(" ")
    print("Calculate R2")
    kMeans_r2 = cluster_obj.get_r2_kmeans()


class performKMeans:
    def __init__(self, data, nClusters=5, init="k-means++"):

        print("Performing K-Means Cluster Analysis")
        print(f"Number of clusters: {nClusters}")
        print(f"Initialization method: {init}")

        self.data = data
        self.nClusters = nClusters
        self.init = init
        self.clusterModel = KMeans(
            n_clusters=nClusters, init=init, n_init=15, random_state=0
        )

    def fitKmeans(self):
        data = self.data
        clusterModel = self.clusterModel
        clusterModel.fit(data)

        return clusterModel

    def performPredictions(self, clusterModel):

        dataOrig = self.data
        dataAssess = dataOrig.copy()
        clusterModel = clusterModel

        clusterPredictions = clusterModel.predict(dataOrig)
        dataAssess["KMeansLabels"] = clusterPredictions

        print("Number of observation per cluster")
        display(dataAssess.groupby("KMeansLabels").count().iloc[:, 0])

        print(" ")
        print("Cluster Means:")
        display(dataAssess.groupby("KMeansLabels").mean())

        print(" ")
        print("Cluster Standard Deviation:")
        display(dataAssess.groupby("KMeansLabels").std())

        # get total variance
        sst = get_ss(dataOrig)
        # compute ssw for each cluster labels
        ssw_labels = dataAssess.groupby(by="KMeansLabels").apply(get_ss)
        # Obtain SSB. Remember: SST = SSW + SSB
        ssb = sst - np.sum(ssw_labels)
        r2 = ssb / sst

        print(f"K-Means R2: {r2}")
        print(" ")
        features_r2 = r2_variables(dataAssess, "KMeansLabels")[:-1]

        print("Evaluating Cluster Performance:")

        metric_evaluation(dataAssess, "KMeansLabels")
        print(" ")

        print("Checking how much variance of each feature is explained by the cluster:")
        print(features_r2)

        clusterProfile = dataAssess.groupby("KMeansLabels").mean()
        multiple_radar_plot(clusterProfile, clusterProfile.columns)

        return dataAssess


class performGMM:
    def __init__(
        self, data, n_components=5, covariance_type="full", init_params="k-means++"
    ):

        from sklearn.mixture import GaussianMixture

        print("Performing GMM Cluster Analysis")
        print(f"Number of clusters: {n_components}")
        print(f"Initialization method: {init_params}")

        self.data = data
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.init_params = init_params
        self.clusterModel = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            n_init=15,
            random_state=0,
        )

    def fit(self):
        data = self.data
        clusterModel = self.clusterModel
        clusterModel.fit(data)
        self.clusterModel_fitted = clusterModel

        return clusterModel

    def performPredictions(self, data=None):

        dataOrig = self.data
        dataAssess = dataOrig.copy()
        clusterModel = self.clusterModel_fitted

        if type(data) == pd.DataFrame:
            dataOrig = data

        col = "GMMLabels"

        clusterPredictions = clusterModel.predict(dataOrig)
        dataAssess[col] = clusterPredictions

        print("Number of observation per cluster")
        display(dataAssess.groupby(col).count().iloc[:, 0])

        print(" ")
        print("Cluster Means:")
        display(dataAssess.groupby(col).mean())

        print(" ")
        print("Cluster Standard Deviation:")
        display(dataAssess.groupby(col).std())

        # get total variance
        sst = get_ss(dataOrig)
        # compute ssw for each cluster labels
        ssw_labels = dataAssess.groupby(by=col).apply(get_ss)
        # Obtain SSB. Remember: SST = SSW + SSB
        ssb = sst - np.sum(ssw_labels)
        r2 = ssb / sst

        print(f"GMM R2: {r2}")
        print(" ")
        features_r2 = r2_variables(dataAssess, col)[:-1]

        print("Evaluating Cluster Performance:")

        metric_evaluation(dataAssess, col)
        print(" ")

        print("Checking how much variance of each feature is explained by the cluster:")
        print(features_r2)

        clusterProfile = dataAssess.groupby(col).mean()
        multiple_radar_plot(clusterProfile, clusterProfile.columns)

        self.assessCluster = dataAssess

        return dataAssess

    def renameClusterLabels(self, nameList):
        dataAssess = self.assessCluster.copy()
        uniqueLabels = list(dataAssess.iloc[:, -1].sort_values().unique())

        col = dataAssess.iloc[:, -1].name

        for labels, newLabels in zip(uniqueLabels, nameList):
            dataAssess.loc[dataAssess[col] == labels, col] = newLabels

        self.assessCluster = dataAssess


def fitSOM(data):
    import sompy
    from sompy.visualization.mapview import View2D
    from sompy.visualization.bmuhits import BmuHitsView
    from sompy.visualization.hitmap import HitMapView

    cols = data.columns
    sm = sompy.SOMFactory().build(
        data.values,
        mapsize=[50, 50],
        initialization="pca",
        training="batch",
        component_names=cols,
    )
    sm.train(n_job=4, verbose="info", train_finetune_len=100, train_rough_len=100)

    return sm


def component_planes(model):
    import sompy
    from sompy.visualization.mapview import View2D
    from sompy.visualization.bmuhits import BmuHitsView
    from sompy.visualization.hitmap import HitMapView

    sns.set()
    view2D = View2D(12, 12, "", text_size=10)
    view2D.show(model, col_sz=3, what="codebook")
    plt.subplots_adjust(top=0.90)
    plt.suptitle("Component Planes", fontsize=20)
    plt.show()


def Umatrix(model):
    import sompy
    from sompy.visualization.mapview import View2D
    from sompy.visualization.bmuhits import BmuHitsView
    from sompy.visualization.hitmap import HitMapView

    u = sompy.umatrix.UMatrixView(
        12, 12, "umatrix", show_axis=True, text_size=8, show_text=True
    )

    UMAT = u.show(
        model,
        distance=1,
        row_normalized=False,
        show_data=True,
        contour=True,  # Visualize isomorphic curves
        blob=False,
    )

    UMAT[1]


def hit_map(model):
    import sompy
    from sompy.visualization.mapview import View2D
    from sompy.visualization.bmuhits import BmuHitsView
    from sompy.visualization.hitmap import HitMapView

    vhts = BmuHitsView(12, 12, "Hits Map")
    vhts.show(model, anotate=True, onlyzeros=False, labelsize=12, cmap="Blues")
    plt.show()


def SOM_Assess(model):
    component_planes(model)
    Umatrix(model)
    hit_map(model)


def som_r2(data, som_model, max_clust):
    sum_r2 = []
    for i in np.arange(1, max_clust):
        print(f"Cluster Nr : {i}")
        som_labels = SOM_Kmeans(data, som_model, i)
        sum_r2.append(calculate_r2(data, som_labels["SOM_KMeans_Label"], "SOM"))

    return sum_r2


# K-Means on top of SOM
def SOM_Kmeans(data, som_model, k):
    from sklearn.neighbors import KNeighborsClassifier
    import sompy
    from sompy.visualization.hitmap import HitMapView

    kmeans = KMeans(n_clusters=k, init="k-means++", n_init=20, random_state=42)
    nodeclus_labels = kmeans.fit_predict(som_model.codebook.matrix)
    som_model.cluster_labels = nodeclus_labels  # setting the cluster labels of sompy

    hits = HitMapView(12, 12, "Clustering", text_size=10)
    hits.show(som_model, anotate=True, onlyzeros=False, labelsize=7, cmap="Pastel1")

    plt.show()
    nodes = som_model.codebook.matrix

    df_nodes = pd.DataFrame(nodes, columns=data.columns)
    df_nodes[f"SOM_KMeans_Label"] = nodeclus_labels

    bmus_map = som_model.find_bmu(data)[0]  # get bmus for each observation in df

    df_bmus = pd.DataFrame(
        np.concatenate((data, np.expand_dims(bmus_map, 1)), axis=1),
        index=data.index,
        columns=np.append(data.columns, "BMU"),
    )
    df_bmus
    df_final = df_bmus.merge(
        df_nodes[f"SOM_KMeans_Label"], "left", left_on="BMU", right_index=True
    )

    return df_final


# Hierarquical Cluster on top of SOM
def SOM_HC(data, som_model, linkage, k):
    # Perform Hierarchical clustering on top of the 2500 untis (sm.get_node_vectors() output)
    hierclust = AgglomerativeClustering(n_clusters=k, linkage=linkage)
    nodeclus_labels = hierclust.fit_predict(som_model.codebook.matrix)
    som_model.cluster_labels = nodeclus_labels  # setting the cluster labels of sompy

    hits = HitMapView(12, 12, "Clustering", text_size=10)
    hits.show(som_model, anotate=True, onlyzeros=False, labelsize=7, cmap="Pastel1")

    plt.show()
    nodes = som_model.codebook.matrix

    df_nodes = pd.DataFrame(nodes, columns=data.columns)
    df_nodes[f"SOM_KMeans_Label"] = nodeclus_labels

    bmus_map = som_model.find_bmu(data)[0]  # get bmus for each observation in df

    df_bmus = pd.DataFrame(
        np.concatenate((data, np.expand_dims(bmus_map, 1)), axis=1),
        index=data.index,
        columns=np.append(data.columns, "BMU"),
    )
    df_bmus
    df_final = df_bmus.merge(
        df_nodes[f"SOM_KMeans_Label"], "left", left_on="BMU", right_index=True
    )


def estimate_Bandwith(data, quantile):
    bandwidth = estimate_bandwidth(data, random_state=1, n_jobs=-1, quantile=quantile)
    return bandwidth


def mean_shift_cluster(data, quantile):
    from sklearn.cluster import MeanShift, DBSCAN, estimate_bandwidth

    bandwidth = estimate_Bandwith(data, quantile)
    print(f"The bandwidth is {bandwidth}")
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, n_jobs=-1).fit(
        df[metric_features]
    )
    ms_labels = ms.predict(df[metric_features])

    ms_n_clusters = len(np.unique(ms_labels))
    print("Number of estimated clusters : %d" % ms_n_clusters)
    data["ms_results"] = ms_labels
    print(data["ms_results"].value_counts())

    return ms
