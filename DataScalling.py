# data scalling
import pandas as pd
import numpy as np


def standardize(x):
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
    return x_scaled


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


def standardize_2(x_train, x_val):
    """""" """""
        x_train - pandas df with features to be normalized (fitted)
        x_val - pandas df with features to be normalized (transfermed)
        
    Receives x_train and x_val and applies standard scalling. The model is fitted only to the training data to avoid leakaage 
    Returnes
        list with x_train and x_val normalized
    """ """""" ""
    x = pd.DataFrame(x_train)
    metric_features = x.select_dtypes(include=np.number).columns
    cat_features = x.select_dtypes(exclude=np.number).columns

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(x[metric_features])
    x_train_scaled = scaler.transform(x[metric_features])  # this will return an array
    # Convert the array to a pandas dataframe
    x_train_scaled = pd.DataFrame(x_train_scaled, columns=metric_features).set_index(
        x.index
    )

    x_val_scaled = pd.DataFrame(
        scaler.transform(x_val[metric_features]),
        index=x_val.index,
        columns=metric_features,
    )
    scaled_data = [
        pd.concat([x_train_scaled, x_train[cat_features]], axis=1),
        pd.concat([x_val_scaled, x_val[cat_features]], axis=1),
    ]
    return scaled_data


def full_standardize(x_train, x_val, test):
    """""" """""
        x_train - pandas df with features to be normalized (fitted)
        x_val - pandas df with features to be normalized (transfermed)
        test - pandas df with features to be normalized (transfermed)
        
    Receives x_train, x_val and a test df and applies standard scalling. The model is fitted only to the training data to avoid leakaage 
    Returnes
        list with x_train,  x_val and test normalized
    """ """""" ""
    metric_features = x_train.select_dtypes(include=np.number).columns
    cat_features = x_train.select_dtypes(exclude=np.number).columns
    x = pd.DataFrame(x_train[metric_features])
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(x)
    x_train_scaled = scaler.transform(x)  # this will return an array
    # Convert the array to a pandas dataframe
    x_train_scaled = pd.DataFrame(x_train_scaled, columns=x.columns).set_index(x.index)
    x_train_scaled = pd.concat([x_train_scaled, x_train[cat_features]], axis=1)

    x_val_scaled = pd.DataFrame(
        scaler.transform(x_val[metric_features]),
        index=x_val.index,
        columns=metric_features,
    )
    x_val_scaled = pd.concat([x_val_scaled, x_val[cat_features]], axis=1)
    test_scaled = pd.DataFrame(
        scaler.transform(test[metric_features]),
        index=test.index,
        columns=metric_features,
    )
    test_scaled = pd.concat([test_scaled, test[cat_features]], axis=1)
    scaled_data = [x_train_scaled, x_val_scaled, test_scaled]
    return scaled_data


def minmax(x):
    """""" """""
        x - pandas df with features to be normalized
        
    Applies minmax normalization to a set of input features 
    Returnes
        scaled df
    """ """""" ""
    metric_features = x.select_dtypes(include=np.number).columns
    cat_features = x.select_dtypes(exclude=np.number).columns
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler().fit(x[metric_features])
    x_scaled = scaler.transform(x[metric_features])  # this will return an array
    # Convert the array to a pandas dataframe
    x_scaled = pd.DataFrame(x_scaled, columns=metric_features).set_index(x.index)
    x_scaled = pd.concat([x_scaled, x[cat_features]], axis=1)
    return x_scaled


def minmax_2(x_train, x_val):
    """""" """""
        x_train - pandas df with features to be normalized (fitted)
        x_val - pandas df with features to be normalized (transfermed)
        
    Receives x_train and x_val and applies minmax scalling. The model is fitted only to the training data to avoid leakaage 
    Returnes
        list with x_train and x_val normalized
    """ """""" ""
    x = pd.DataFrame(x_train)
    metric_features = x.select_dtypes(include=np.number).columns
    cat_features = x.select_dtypes(exclude=np.number).columns

    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler().fit(x[metric_features])
    x_train_scaled = scaler.transform(x[metric_features])  # this will return an array
    # Convert the array to a pandas dataframe
    x_train_scaled = pd.DataFrame(x_train_scaled, columns=metric_features).set_index(
        x.index
    )

    x_val_scaled = pd.DataFrame(
        scaler.transform(x_val[metric_features]),
        index=x_val.index,
        columns=metric_features,
    )
    scaled_data = [
        pd.concat([x_train_scaled, x_train[cat_features]], axis=1),
        pd.concat([x_val_scaled, x_val[cat_features]], axis=1),
    ]
    return scaled_data


def full_minmax(x_train, x_val, test):
    """""" """""
        x_train - pandas df with features to be normalized (fitted)
        x_val - pandas df with features to be normalized (transfermed)
        test - pandas df with features to be normalized (transfermed)
        
    Receives x_train, x_val and a test df and applies minmax scalling. The model is fitted only to the training data to avoid leakaage 
    Returnes
        list with x_train,  x_val and test normalized
    """ """""" ""
    metric_features = x_train.select_dtypes(include=np.number).columns
    cat_features = x_train.select_dtypes(exclude=np.number).columns
    x = pd.DataFrame(x_train[metric_features])
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler().fit(x)
    x_train_scaled = scaler.transform(x)  # this will return an array
    # Convert the array to a pandas dataframe
    x_train_scaled = pd.DataFrame(x_train_scaled, columns=x.columns).set_index(x.index)
    x_train_scaled = pd.concat([x_train_scaled, x_train[cat_features]], axis=1)

    x_val_scaled = pd.DataFrame(
        scaler.transform(x_val[metric_features]),
        index=x_val.index,
        columns=metric_features,
    )
    x_val_scaled = pd.concat([x_val_scaled, x_val[cat_features]], axis=1)
    test_scaled = pd.DataFrame(
        scaler.transform(test[metric_features]),
        index=test.index,
        columns=metric_features,
    )
    test_scaled = pd.concat([test_scaled, test[cat_features]], axis=1)
    scaled_data = [x_train_scaled, x_val_scaled, test_scaled]
    return scaled_data


def MinMax(x):
    """""" """""
        x - pandas df with features to be normalized
        
    Applies minmax normalization to a set of input features 
    Returnes
        scaled df
    """ """""" ""
    metric_features = x.select_dtypes(include=np.number).columns
    cat_features = x.select_dtypes(exclude=np.number).columns
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler().fit(x[metric_features])
    x_scaled = scaler.transform(x[metric_features])  # this will return an array
    # Convert the array to a pandas dataframe
    x_scaled = pd.DataFrame(x_scaled, columns=metric_features).set_index(x.index)
    x_scaled = pd.concat([x_scaled, x[cat_features]], axis=1)
    return (x_scaled, scaler)
