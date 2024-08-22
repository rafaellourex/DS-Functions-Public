# OUTLIERS
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def formatColumns(df):
    # replace column names
    for col in df.columns:
        new_name = col.replace(" ", "_").lower()
        df = df.rename(columns={col: new_name})

    return df


def BasicPreparation(df):
    import pandas as pd
    import numpy as np

    col_list = []
    values_list = []
    diff_0 = []
    data = df.copy()
    # removing columns due to redundant data (having all the same value)
    for col in df.select_dtypes(include=np.number).columns:
        mean = df[col].mean()
        max_ = df[col].max()
        min_ = df[col].min()

        if mean == max_ == min_:
            if mean != 0:
                diff_0.append(col)
            df = df.drop(columns=col)
            col_list.append(col)
            values_list.append(mean)
    print(f"{len(col_list)} were removed due to redundant data.")
    print(f"Values: {set(values_list)}")

    # removing columns that have more than 9% of the data missing
    mv_ = (df.isnull().sum() / len(df)).sort_values(ascending=False)
    col_list_2 = []
    for col, value in zip(mv_.index, mv_.values):
        if value >= 0.9:
            col_list_2.append(col)

    df = df.drop(columns=col_list_2)

    col_list_3 = []
    for col in df.select_dtypes(exclude=np.number):
        value = len(df[col].value_counts().index)
        if value == 1:
            col_list_3.append(col)

    df = df.drop(columns=col_list_3)

    print(" ")
    print(
        f"{len(col_list_2)} were removed due to missing values & {len(col_list_3)} dua to redundant categorical features."
    )
    print(" ")
    total_cols = len(col_list) + len(col_list_2) + len(col_list_3)
    print(f"{total_cols} columns were removed in total.")
    print(
        f"{total_cols/len(data.columns)}% of the columns were removed due to redundant data"
    )
    print(" ")
    print(f"The new df has {len(df.columns)} columns")

    df = formatColumns(df)

    return df


def get_description(data):
    nr_rows = data.shape[0]
    nr_cols = data.shape[1]

    num_cols = data.select_dtypes(exclude=np.object).columns
    cat_cols = data.select_dtypes(include=np.object).columns

    print(data.info())
    print(" ")
    display(data.describe().T)

    print(f"Nr of rows: {nr_rows}")
    print(f"Nr of columns: {nr_cols}")
    print(" ")
    print("Missing Values:")
    missing = data.isnull().sum().sort_values(ascending=False)
    display(missing)

    print("Missing Values, %:")
    display(missing / len(data))

    print(f"Nr of numerical features: {len(num_cols)}")
    print(f"Nr of categorical features: {len(cat_cols)}")

    if len(num_cols) < 30:
        print("Numerical Features:")
        print(num_cols)

    print(" ")
    if len(cat_cols) < 30:
        print("Categorical Features:")
        print(cat_cols)

    return (num_cols, cat_cols)


# analyze skewness
def skewFix(df):
    import numpy as np
    from scipy.stats import skew

    def replace_inf_with_zero(df):

        # Replace infinite values with 0
        df.replace([np.inf, -np.inf], 0, inplace=True)
        return df

    columns = df.columns
    df_transform = df.copy()

    transformations_ = []
    for col in columns:
        transform = dict()
        if df_transform[col].dtype == np.number:
            sk = skew(df_transform[col])
            min_ = df_transform[col].min()
            if (sk > 1.5) & (min_ > 0):
                df_transform[col] = np.log(df_transform[col])
                transform["transformation"] = "Log"
                transform["feature"] = col
                transformations_.append(transform)

            if (sk > 1.5) & (min_ <= 0):
                df_transform[col] = np.cbrt(df_transform[col])
                transform["transformation"] = "CBRoot"
                transform["feature"] = col
                transformations_.append(transform)

    df_transform = replace_inf_with_zero(df_transform)
    transform_df = pd.DataFrame.from_records(transformations_)
    return (df_transform, transform_df)


def skewTest(df):
    import numpy as np
    from scipy.stats import skew

    sk = skew(df)

    df_skew = pd.DataFrame(
        data=sk,
        index=df.columns,
    )
    return df_skew


def analyzeQuantiles(data):
    """""" """""" """"
    
    Receives:
            pandas dataframe
            
    Creates a df that analyzes outliers by creating 2 disparity features that measure the level of the outliers.
    The fist measure divides de 95% quantile by the 90% and the second divides the mean by the median
    
    Note: thresholds can be fyrther adjusted
    
    """ """""" """""" ""
    df_out = data

    quantile_disp = np.round(df_out.describe([0.25, 0.5, 0.75, 0.9, 0.95]).T).iloc[
        :, :-1
    ]
    quantile_disp["disparity"] = (quantile_disp["95%"] / quantile_disp["90%"]).replace(
        np.inf, 0
    )
    quantile_disp["mean_disp"] = (quantile_disp["mean"] / quantile_disp["50%"]).replace(
        np.inf, 0
    )

    quantile_disp = quantile_disp.sort_values(ascending=False, by="mean_disp")
    display(quantile_disp.iloc[:10])

    return quantile_disp


def replaceOutliers(data):
    """""" """""" """"
    
    Receives:
            pandas dataframe
            
    Replaces outliers with the nearest quantile, per example an observation of quantile 99 will be replaced by the quantile 90
    
    Note: thresholds can be fyrther adjusted
    
    """ """""" """""" ""
    df_out = data.copy()

    cols = df_out.select_dtypes(include=np.number).columns
    for col in cols:
        upper_quant = df_out[col].quantile(0.85)
        lower_quant = df_out[col].quantile(0.15)

        to_replace_upper = df_out[col].quantile(0.9)
        to_replace_lower = df_out[col].quantile(0.1)

        df_out.loc[df_out[col] > to_replace_upper, col] = upper_quant
        df_out.loc[df_out[col] < to_replace_lower, col] = lower_quant

    return df_out


def mean_varince_cutoff(df, remove_pct, thresh):
    """""" """""" """
    it receives a df, a % removal limit and a treshold for how strict the cutt of limit must be 
    Note: threshold of 2 means 2 stds from the mean 
    
    remove_pct: Range- [0,1], if == 0.03 then around 3% of the df will be removed
    """ ""
    import time

    start_time = time.time()
    df = df.copy()
    remove_pct = remove_pct
    thresh = thresh

    len_ = len(df)
    max_remove = len(df) - (len(df) * remove_pct)

    col_index = (
        abs(
            df.select_dtypes(include=np.number).max()
            / df.select_dtypes(include=np.number).mean()
        )
        .sort_values(ascending=False)
        .index
    )

    for col in col_index:
        if len(df) >= max_remove:
            avg_ = df[col].mean()
            std = df[col].std()
            upper_lim = avg_ + std * thresh
            lower_lim = avg_ - std * thresh

            df = df.loc[(df[col] >= lower_lim) & (df[col] <= upper_lim)]
        else:
            final_len = len(df) / len_

            print(f"Cut off level: {remove_pct}")
            print(f"std treshold: {thresh}")
            print(" ")
            print(f"{final_len}% of the df remained")
            print(" ")
            print("Removing oultiers lasted:")
            print("--- %s seconds ---" % (time.time() - start_time))
            print(" ")
            return df


def IQR(df, threshold):
    """""" """"
    Receives:
        pandas df 
        threshold - sensitivity threshold to eliminate outliers 
    Eliminates every data point that excedes the thresold
    
    """ """""" ""

    lenght = df.shape[0]
    data = df.copy()
    metric_features = list(
        data.select_dtypes(include=np.number).set_index(data.index).columns
    )
    q1 = df.quantile(q=0.25)
    q3 = df.quantile(q=0.75)
    iqr = q3 - q1
    lower_lim = q1 - threshold * iqr
    upper_lim = q3 + threshold * iqr
    filters = []
    for metric in metric_features:
        llim = lower_lim[metric]
        ulim = upper_lim[metric]
        filters.append(df[metric].between(llim, ulim, inclusive=True))

    df = df[np.all(filters, 0)]
    final_lenght = df.shape[0]
    print(
        "Percentage of data kept after removing outliers:",
        np.round(final_lenght / lenght, 4),
    )
    return df


def standardize(x):
    x = pd.DataFrame(x)
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(x)
    x_scaled = scaler.transform(x)  # this will return an array
    # Convert the array to a pandas dataframe
    x_scaled = pd.DataFrame(x_scaled, columns=x.columns).set_index(x.index)
    return x_scaled


def K_means_outliers(data, n_cluster=25):
    metric_features = list(
        data.select_dtypes(include=np.number).set_index(data.index).columns
    )
    from sklearn.cluster import KMeans

    init_len = len(data)
    data_kmeans = standardize(data.loc[:, metric_features].copy())
    min_len = np.round(len(data_kmeans) * 0.95, 0)

    while len(data_kmeans) > min_len:
        outlier_kmeans = KMeans(
            n_clusters=n_cluster, n_init="auto", init="k-means++", random_state=0
        )
        fit = outlier_kmeans.fit(data_kmeans)
        data_kmeans["clusters"] = fit.predict(data_kmeans)
        clusters_ = (
            data_kmeans["clusters"]
            .value_counts()
            .sort_values(ascending=False)[:-1]
            .index
        )

        data_kmeans = data_kmeans.loc[data_kmeans["clusters"].isin(clusters_)]

    outliers_kmeans = pd.DataFrame(index=data.index)
    outliers_kmeans.loc[outliers_kmeans.index.isin(data_kmeans.index), "kmeans"] = 1
    outliers_kmeans.loc[
        outliers_kmeans.index.isin(data_kmeans.index) == False, "kmeans"
    ] = 0

    df_return = data.loc[data.index.isin(data_kmeans.index)]

    fin_len = len(df_return)
    pct_removed = (1 - (fin_len / init_len)) * 100
    print(" ")
    print(f"The new datset has {len(data_kmeans)} observations")
    print(f"{pct_removed}% of the data was removed")
    return (df_return, outliers_kmeans)


def KMeans_Outliers_Algorithm(df):
    # set metric columns
    metric_cols = df.select_dtypes(include=np.number).columns
    # set data to model
    X = standardize(df[metric_cols].copy())

    # specify the nr of observations to remove at each iteration
    to_remove = len(X) // 100
    if to_remove < 1:
        to_remove = 10

    # specify model, nr of clusters must be one in this case
    model = KMeans(n_clusters=1, n_init="auto", init="k-means++", random_state=0)
    model.fit(X)

    # get distances bwtween each observation and the centroid
    distances = model.transform(X)

    # create dataframe with distances
    distances_df = pd.DataFrame(index=X.index)
    distances_df["distances"] = distances

    # sort ditances from higher to lower abd choose observations to remove
    to_remove = distances_df.sort_values(by="distances", ascending=False).iloc[
        :to_remove
    ]

    X_out = df.loc[df.index.isin(to_remove.index) == False]
    return X_out


def KMeans_outliers(df):
    thresh = 0.05
    df_out = df.copy()
    init_len = len(df_out)
    pct_removed = 1

    while pct_removed > ((1 - thresh)):
        df_out = KMeans_Outliers_Algorithm(df_out)
        pct_removed = np.round(len(df_out) / init_len, 4)

    df_return = df.loc[df.index.isin(df_out.index)]

    print(f"{pct_removed}% of the data was removed from the dataset")
    return df_return


# Isolation Forest
def Isolation_Forest(data, contamination):
    import time

    start_time = time.time()

    # this algorithm only works with data without missing values
    from sklearn.ensemble import IsolationForest

    # data_out = knn_imput(data.select_dtypes(include=np.number).set_index(data.index))
    data_out = data.select_dtypes(include=np.number).set_index(data.index)
    forest_model = IsolationForest(
        n_estimators=100, warm_start=True, contamination=contamination, random_state=0
    )
    forest_model.fit(data_out)
    anomally = forest_model.decision_function(data_out)
    predict_outcome = forest_model.predict(data_out)

    # creating a dataset with the density scores and the predicted outcome (-1==outlier; 1==normal obs)
    data_out_score = pd.DataFrame(data=anomally, index=data_out.index)
    data_out_score["predicted_outcome"] = predict_outcome
    index = data_out_score.loc[data_out_score["predicted_outcome"] == 1].index

    data_return = data.loc[data.index.isin(index)]

    print("Isolation Forest lasted:")
    print("--- %s seconds ---" % (time.time() - start_time))
    print(" ")
    return (data_out_score, data_return)


def Isolation_Forest_test(data):
    outliers = Isolation_Forest(data)
    outliers_test = pd.DataFrame(index=data.index)
    outliers_test["predicted_outcome"] = outliers["predicted_outcome"]
    outliers_test.loc[outliers_test["predicted_outcome"] == 1, "predicted_outcome"] = 0
    outliers_test.loc[outliers_test["predicted_outcome"] == -1, "predicted_outcome"] = 1
    return outliers_test


def lof_outliers(df, n_neighbors=10, contamination="auto"):

    import numpy as np
    from sklearn.neighbors import LocalOutlierFactor

    df_out_assess = pd.DataFrame(index=df.index)
    df_model = df.select_dtypes(include=np.number).copy()
    model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    outliers = model.fit_predict(df_model.select_dtypes(include=np.number))
    scores = model.negative_outlier_factor_
    # outliers = model.predict(df)
    df_out_assess["scores"] = scores
    df_out_assess["outliers"] = outliers

    index_ = df_out_assess.loc[df_out_assess.outliers == 1].index
    df_out = df.loc[df.index.isin(index_)]

    init_len = len(df)
    fin_len = len(df_out)
    pct_rem = np.round(1 - (fin_len / init_len), 3) * 100
    nr_rem = init_len - fin_len
    print(f"# of observations removed: {nr_rem}")
    print(f"% of observations removed: {pct_rem}%")
    return (df_out, df_out_assess, model)


# winsorizing
def Winsorizing(df):
    df_smooth = df.copy()

    for col in df_smooth.select_dtypes(include=np.number).columns:
        lower_bound = df_smooth[col].quantile(0.01)
        higher_bound = df_smooth[col].quantile(0.99)
        df_smooth.loc[df_smooth[col] > higher_bound, col] = higher_bound
        df_smooth.loc[df_smooth[col] < lower_bound, col] = lower_bound
    return df_smooth


# MISSING VALUES
# knn imputer
def knn_imput(data, k=5, weights="distance", metric="nan_euclidean"):
    """""" """
    Receives
        pandas df to imput missing values
        k - number of neighbours used in KNN, predefined is 5
    
    """ """""" ""
    from sklearn.impute import KNNImputer

    print(f"Performing KNN Imputation:")
    print(f"K = {k}")
    print(f"Weights = {weights}")
    print(f"Metric = {metric}")
    print(" ")

    metric_features = data.select_dtypes(include=np.number).columns
    cat_features = data.select_dtypes(exclude=np.number).columns

    imputer = KNNImputer(n_neighbors=k, weights=weights, metric=metric)

    imputer.fit(data[metric_features])
    transform = imputer.transform(data[metric_features])

    x = pd.DataFrame(transform, columns=metric_features, index=data.index)

    x = pd.concat([x, data[cat_features]], axis=1)
    return x


def interactiveImputer(data, estimator=None, sample=10000):
    """""" """
    Receives
        pandas df to imput missing values
        estimator - model that will be used in the iteractive imputer 
            if estimator==None then Decision Tree will be used as estimator
    
    """ """""" ""

    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import SimpleImputer
    from sklearn.impute import IterativeImputer

    if estimator == None:
        from sklearn.tree import DecisionTreeRegressor

        estimator = DecisionTreeRegressor(random_state=0, max_depth=4)

    print(f"Base estimator: {estimator}")

    metric_features = data.select_dtypes(exclude=np.object).columns
    cat_features = data.select_dtypes(include=np.object).columns

    imputer = IterativeImputer(estimator=estimator)
    imputer.fit(data.sample(sample)[metric_features])
    data_imputed = imputer.transform(data[metric_features])
    data_imputed = pd.DataFrame(data_imputed, columns=metric_features, index=data.index)

    data_return = pd.concat([data_imputed, data[cat_features]], axis=1)

    return (data_return, imputer)


def split_regressor(x, y):
    from sklearn.model_selection import train_test_split

    X_train, X_val, y_train, y_val = train_test_split(
        x, y, test_size=0.3, random_state=0, shuffle=True
    )
    return (X_train, y_train, X_val, y_val)


def interactiveImputer_comparison(data, target, estimators, model):
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import SimpleImputer
    from sklearn.impute import IterativeImputer

    metric_features = (
        data.drop(columns=f"{target}").select_dtypes(exclude=np.object).columns
    )
    cat_features = data.select_dtypes(include=np.object).columns
    scores_train = []
    scores_val = []
    models = []

    for estimator in estimators:
        print(f"Testing: {estimator}")
        imputer = IterativeImputer(estimator=estimator)
        imputer.fit(data[metric_features])
        data_imputed = imputer.transform(data[metric_features])
        data_imputed = pd.DataFrame(
            data_imputed, columns=metric_features, index=data.index
        )

        x = data_imputed
        y = data[f"{target}"]

        x_train, y_train, x_val, y_val = split_regressor(x, y)
        model.fit(x_train, y_train)

        scores_train.append(model.score(x_train, y_train))
        scores_val.append(model.score(x_val, y_val))
        models.append(estimator)

    report = pd.DataFrame(index=models)
    report["Train Accuracy"] = scores_train
    report["Val Accuracy"] = scores_val

    full_data = pd.concat([data_imputed[metric_features], data[cat_features]], axis=1)

    return (full_data, report)
