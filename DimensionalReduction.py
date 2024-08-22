import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pylab as plt


def Standardize(x):
    """""" """""
        x - pandas df with features to be normalized
        
    Standardizes a set of input features (mean= 0,var = 1)
    Returnes
        scaled df
    """ """""" ""
    from sklearn.preprocessing import StandardScaler

    metric_features = x.select_dtypes(include=np.number).columns
    cat_features = x.select_dtypes(exclude=np.number).columns
    scaler = StandardScaler().fit(x[metric_features])
    x_scaled = scaler.transform(x[metric_features])  # this will return an array
    # Convert the array to a pandas dataframe
    x_scaled = pd.DataFrame(x_scaled, columns=metric_features).set_index(x.index)
    x_scaled = pd.concat([x_scaled, x[cat_features]], axis=1)
    return (x_scaled, scaler)


def PCA_Assess_loadings(data, components, n_components):
    df = pd.concat([data, components], axis=1)
    loadings = df.corr().iloc[:-n_components, -n_components:]

    def _color_red_or_green(val):
        if val < -0.45:
            color = "background-color: red"
        elif val > 0.45:
            color = "background-color: green"
        else:
            color = ""
        return color

    return loadings.style.applymap(_color_red_or_green)


class performPCA:

    def __init__(self, data, n_components=5, scalling=True):

        from sklearn.decomposition import PCA as PCA

        print("Performing PCA")
        print(f"Number of components to retain: {n_components}")

        metric_features = list(
            data.select_dtypes(include=np.number).set_index(data.index).columns
        )
        self.data = data[metric_features]
        self.n_components = n_components
        self.PCA_Model = PCA(n_components=n_components, random_state=0)
        self.scalling = scalling
        if scalling == True:
            scaled_df, scaler = Standardize(self.data)
            self.data_scaled = scaled_df
            self.scaler = scaler

    def fit(self):
        model = self.PCA_Model

        if self.scalling == True:
            print("Scalling will be performed - Z-Score Normalization")
            data = self.data_scaled
        else:
            print("Scalling will not be performed")
            data = self.data

        self.dataFit = data

        model_fitted = model.fit(data)
        self.model_fitted = model_fitted

        components = pd.DataFrame(model_fitted.transform(data), index=data.index)
        for i in components.columns:
            components.rename(columns={i: f"PC_{i}"}, inplace=True)
        self.components = components

        return model_fitted

    def assessPCA(self):

        pca_fitted = self.model_fitted
        df = pd.DataFrame(
            {
                "Eigenvalue": pca_fitted.explained_variance_,
                "Difference": np.insert(np.diff(pca_fitted.explained_variance_), 0, 0),
                "Proportion": pca_fitted.explained_variance_ratio_,
                "Cumulative": np.cumsum(pca_fitted.explained_variance_ratio_),
            },
            index=range(1, pca_fitted.n_components_ + 1),
        )

        cum_sum = pca_fitted.explained_variance_ratio_.cumsum()
        exp_var = pca_fitted.explained_variance_
        cov = pca_fitted.get_covariance()

        # plot PCA report
        plt.figure(figsize=(20, 10))
        ax1 = plt.subplot(1, 2, 1)
        title = "Eigenvalues of each component"
        sns.lineplot(data=df.loc[:, ["Eigenvalue"]], ax=ax1)
        plt.axhline(1, ls="--")
        ax1.set_title(title, size=15, weight="bold")
        ax1.set_ylabel("Eigenvalue", size=15)
        plt.xticks(size=15)
        plt.yticks(size=15)
        ax1.set_xlabel("Nr of Pricipal Components", size=15, weight="bold")

        ax2 = plt.subplot(1, 2, 2)
        title = "Cumulative % of total variance explained by the components"
        sns.lineplot(data=df.loc[:, ["Cumulative", "Proportion"]], ax=ax2)
        ax2.set_title(title, size=15, weight="bold")
        ax2.set_ylabel("Cumulative %", size=15)
        plt.xticks(size=15)
        plt.yticks(size=15)
        ax2.set_xlabel("Nr of Pricipal Components", size=15, weight="bold")
        plt.tight_layout()
        full_title = "PCA_Report"
        plt.show()

        print("The variance explained by each component is: " + str(exp_var))
        print(
            "The total variance explained by the components is: "
            + str(sum(pca_fitted.explained_variance_ratio_))
        )
        return df

    def assessLoadings(self):

        data = self.dataFit
        components = self.components
        n_components = self.n_components
        loadings = PCA_Assess_loadings(data, components, n_components)

        self.loadings = loadings
        return loadings

    def renameComponents(self, nameList):
        components = self.components

        cols = components.columns

        if len(cols) == len(nameList):
            for col, newCol in zip(components.columns, nameList):
                components.rename(columns={col: newCol}, inplace=True)


"""
How to use the PCA Class

n_components = 3
pca =  dimRed.performPCA(data= df_eng,
                        n_components=n_components,
                        scalling=True)
pca.fit()
pca_fitted = pca.model_fitted
pca.assessPCA()
pca.assessLoadings()
data_pca = pca.components


"""


def PLS(data, y_name, indexes, n_components):
    from sklearn.cross_decomposition import PLSRegression
    import matplotlib.pyplot as plt
    import seaborn as sns

    data.reset_index()
    metric_features = list(
        data.select_dtypes(include=np.number).set_index(data.index).columns
    )
    x = data.loc[:, metric_features].drop(columns=[f"{y_name}"])

    print(f"Nr of input features for PLS: {len(metric_features)}")
    y = data[f"{y_name}"]
    # x = StandardScaler().fit_transform(x)
    s = pd.DataFrame(x, columns=x.columns)
    s = s.set_index(data.index)
    display(s)
    s = s

    pls = PLSRegression(n_components=n_components)
    transformed_data = pls.fit_transform(X=s, y=y)

    components = pd.DataFrame(transformed_data[0], index=data.index)
    df = pd.DataFrame(
        {"PLS_Component": np.arange(1, n_components + 1)},
        index=range(1, n_components + 1),
    )

    plt.figure(figsize=(20, 10))
    ax1 = plt.subplot(1, 2, 1)
    title = "PLS Components"
    sns.lineplot(data=df, ax=ax1)
    ax1.set_title(title, size=15, weight="bold")
    ax1.set_ylabel("PLS Component", size=15)
    plt.xticks(size=15)
    plt.yticks(size=15)
    ax1.set_xlabel("Nr of PLS Components", size=15, weight="bold")
    full_title = "PLS_Report"
    plt.show()

    print("The variance explained by each PLS component is: " + str(pls.x_scores_))
    print(
        "The total variance explained by the PLS components is: " + str(pls.score(x, y))
    )

    for i in components.columns:
        components.rename(columns={i: f"PLS_{i}"}, inplace=True)

    return df, components, pls


def umap(x_train, n_neighbors, min_dist):
    import umap.umap_ as umap

    # import umap.plot
    reducer = umap.UMAP(
        n_neighbors=n_neighbors, n_components=2, min_dist=min_dist, random_state=0
    )
    mapper = reducer.fit(x_train)
    mapper.fit_transform(x_train)

    embeddings = mapper.embedding_
    return (mapper, embeddings)
