# filter methods
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def choosefeatures(x_train, y_train, model):
    from sklearn.feature_selection import SelectFromModel

    clf = model
    clf = clf.fit(x_train, y_train)
    clf.feature_importances_
    plot_data = pd.DataFrame(
        index=x_train.columns, data=clf.feature_importances_, columns={"Ft_importance"}
    )
    plot_query = plot_data.loc[plot_data["Ft_importance"] > 0.005].copy()
    print(plot_query.index)
    plt.barh(y=plot_query.index, width=plot_query["Ft_importance"])

    model = SelectFromModel(clf, prefit=True, max_features=10)
    feature_idx = model.get_support()

    feature_name = x_train.columns[feature_idx]

    X_new = pd.DataFrame(
        model.transform(x_train), index=x_train.index, columns=feature_name
    )
    return X_new


# F Classification
def F_classification(x, y, n_best):
    """""" """""" """
    Performs F selectiobn selection  
    Receives:
        x and y 
        train,test and gap sizes indicating how the splits are going to be created
        nr_best, representing the nr of independent variables to keep in the selection
        date_index, binary variable indicating if datetime will be used as spliting criteria or if it's the index of x
    Note: for classification problems
    
    """ """""" """"""
    from sklearn.feature_selection import f_classif

    threshold = n_best  # the number of most relevant features
    high_score_features1 = []
    feature_scores = f_classif(x, y)[0]

    mi_scores = pd.DataFrame()
    mi_scores["Coef"] = feature_scores
    mi_scores["features"] = x.columns
    mi_scores = mi_scores.sort_values(ascending=False, by="Coef").iloc[:n_best]
    display(mi_scores.set_index("features"))
    print(list(mi_scores.features))
    return mi_scores.features


# mutual information ranking between independent variables x and a continuos dependent variable y


# for classification
def mutual_information_classification(x, y, n_best):
    """""" """""" """
    Performs mutual information selection using 
    Receives:
        x and y 
        nr_best, representing the nr of independent variables to keep in the selection
    Note: use in regression problems
    
    """ """""" """"""
    import time

    start_time = time.time()
    from sklearn.feature_selection import mutual_info_classif

    n_best = n_best
    scores = mutual_info_classif(x, y)
    mi_scores = pd.DataFrame()
    mi_scores["Coef"] = scores
    mi_scores["features"] = x.columns
    mi_scores = mi_scores.sort_values(ascending=False, by="Coef")[:n_best]

    print(f"Top {n_best} variables according to mutual information:")
    print(mi_scores.features.unique())
    print(" ")
    print("Mutual Information Selection lasted:")
    print("--- %s seconds ---" % (time.time() - start_time))
    return mi_scores


# for regression problems
def mutual_information_regression(x, y, n_best):
    """""" """""" """
    Performs mutual information selection using 
    Receives:
        x and y 
        nr_best, representing the nr of independent variables to keep in the selection
    Note: use in regression problems
    
    """ """""" """"""
    import time

    start_time = time.time()
    from sklearn.feature_selection import mutual_info_regression

    n_best = n_best
    scores = mutual_info_regression(x, y)
    mi_scores = pd.DataFrame()
    mi_scores["Coef"] = scores
    mi_scores["features"] = x.columns
    mi_scores = mi_scores.sort_values(ascending=False, by="Coef")[:n_best]

    print(f"Top {n_best} variables according to mutual information:")
    print(mi_scores.features.unique())
    print(" ")
    print("Mutual Information Selection lasted:")
    print("--- %s seconds ---" % (time.time() - start_time))
    return mi_scores


# this function receives a DF x (composed by numerical values and with index) and returns the most important variables following a Lasso regression
# the dependent variable must be continuos, otherwise we should use a Logistic regression
def select_from_model(x, y, model):
    from sklearn.model_selection import train_test_split
    from sklearn.feature_selection import SelectFromModel
    from sklearn.preprocessing import MinMaxScaler

    # Min_Max = MinMaxScaler()
    # col = x.columns
    # x = pd.DataFrame(Min_Max.fit_transform(x), columns= col)
    # y = Min_Max.fit_transform(y)

    X_train, X_validation, y_train, y_validation = train_test_split(
        x, y, train_size=0.7, shuffle=True, random_state=42
    )

    sel_ = SelectFromModel(model)
    sel_.fit(X_train, np.ravel(y_train, order="C"))
    sel_.get_support()

    X_train = pd.DataFrame(X_train, columns=x.columns)
    selected_feat = X_train.columns[(sel_.get_support())]
    x_columns = X_train.loc[:, selected_feat]
    print("total features: {}".format((X_train.shape[1])))
    print("selected features: {}".format(len(selected_feat)))
    print(
        "features with coefficients shrank to zero: {}".format(
            X_train.shape[1] - len(selected_feat)
        )
    )
    return x_columns.columns


# Wrapper Methods
def RFECV_classifier(x, y, n_best, model=None):
    from sklearn.feature_selection import RFECV
    from sklearn.metrics import (
        r2_score,
        mean_absolute_error,
        mean_squared_error,
        median_absolute_error,
    )
    from sklearn.model_selection import cross_validate
    from sklearn.model_selection import StratifiedKFold

    if model == None:
        from sklearn.tree import DecisionTreeClassifier

        model = DecisionTreeClassifier(max_depth=6, random_state=0)
    cv = StratifiedKFold(n_splits=10)

    cv_ = cross_validate(model, x, y, cv=cv, scoring=["f1", "recall", "precision"])
    f1_mean = np.mean(cv_["test_f1"]).round(2)
    prec_mean = np.mean(cv_["test_precision"]).round(2)
    rec_mean = np.mean(cv_["test_recall"]).round(2)
    print(f"The test F1 score is : {f1_mean}")
    print(f"The test Precision is : {prec_mean}")
    print(f"The test Recall is : {rec_mean}")
    threshold = n_best  # the number of most relevant features

    selector = RFECV(estimator=model, cv=cv, min_features_to_select=1)
    selector_x = selector.fit_transform(x, y)
    selector_ind = selector.get_support()

    selector.ranking_
    selected_features = pd.Series(selector.support_, index=x.columns)
    selected_features = selected_features.sort_values(ascending=False).iloc[:n_best]
    return selected_features


def RFECV_regressor(x, y, n_best, model=None):
    from sklearn.feature_selection import RFECV
    from sklearn.metrics import (
        r2_score,
        mean_absolute_error,
        mean_squared_error,
        median_absolute_error,
    )
    from sklearn.model_selection import cross_validate
    from sklearn.model_selection import KFold

    if model == None:
        from sklearn.tree import DecisionTreeRegressor

        model = DecisionTreeRegressor(max_depth=6, random_state=0)

    cv = KFold(n_splits=10)
    cv_ = cross_validate(model, x, y, cv=cv, scoring=["neg_root_mean_squared_error"])
    test_mean = np.mean(cv_["test_neg_root_mean_squared_error"] * -1).round(2)
    print(f"The test RMSE is {test_mean}")
    threshold = n_best  # the number of most relevant features

    selector = RFECV(estimator=model, cv=cv, min_features_to_select=1)
    selector_x = selector.fit_transform(x, y)
    selector_ind = selector.get_support()

    selector.ranking_
    selected_features = pd.Series(selector.support_, index=x.columns)
    selected_features = selected_features.sort_values(ascending=False).iloc[:n_best]
    return selected_features


# this RFE only works with categorical variables
def RFE_classifier(x, y, n_best, model=None):
    from sklearn.feature_selection import RFE
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    if model is None:
        model = RandomForestClassifier()  # Default model if not provided

    X_train, X_validation, y_train, y_validation = train_test_split(
        x, y, train_size=0.7, shuffle=True, stratify=y, random_state=0
    )

    model.fit(X_train, y_train)

    print(f"The validation score is {model.score(X_validation, y_validation)}")
    print(f"The train score is {model.score(X_train, y_train)}")

    selector = RFE(estimator=model, n_features_to_select=n_best)
    selector_x = selector.fit_transform(X_train, y_train)
    selector_ind = selector.get_support()
    selector.ranking_

    selected_features = pd.Series(selector.support_, index=X_train.columns)
    selected_features = selected_features.sort_values(ascending=False).iloc[:n_best]

    return selected_features


# RFE for regression problems
def RFE_regressor(x, y, n_best, model=None):
    from sklearn.feature_selection import RFECV
    from sklearn.metrics import mean_squared_error

    if model is None:
        model = DecisionTreeRegressor(
            max_depth=6, random_state=0
        )  # Default model if not provided

    X_train, X_validation, y_train, y_validation = train_test_split(
        x, y, train_size=0.7
    )

    model.fit(X_train, y_train)

    rmse_train = np.sqrt(mean_squared_error(y_train, model.predict(X_train)))
    rmse_test = np.sqrt(mean_squared_error(y_validation, model.predict(X_validation)))

    print(f"The train score is {rmse_train}")
    print(f"The validation score is {rmse_test}")

    selector = RFE(estimator=model, n_features_to_select=n_best)
    selector_x = selector.fit_transform(X_train, y_train)
    selector_ind = selector.get_support()
    selector.ranking_

    selected_features = pd.Series(selector.support_, index=X_train.columns)
    selected_features = selected_features.sort_values(ascending=False).iloc[:n_best]

    return selected_features


# forward and backard selections
def forward_selection(x, y, n_best):
    """""" """""" """
    Performs forward selection using a decision tree using stratkfold as validation method
    Receives:
        x and y 
        train_size, test_size and gap 
        k indicating the nr of folds, predefined to 10 
        n_best indicating the nr of features to select
        date_index, binary variable indicating if datetime will be used as spliting criteria or if it's the index of x
        
    Returns:
        n_best features
    
    """ """""" """"""
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.feature_selection import SequentialFeatureSelector
    from sklearn.model_selection import StratifiedKFold

    model = DecisionTreeClassifier(max_depth=6, random_state=0)

    skf = StratifiedKFold(
        n_splits=5,
    )

    forward_selection = SequentialFeatureSelector(
        model,
        direction="forward",
        cv=skf,
        n_features_to_select=n_best,
    )
    forward_selection = forward_selection.fit(x, y)

    subset = x.columns[forward_selection.get_support()]
    return (forward_selection, subset)


def backwards_selection(x, y, n_best):
    """""" """""" """
    Performs forward selection using a decision tree using stratkfold as validation method
    Receives:
        x and y 
        train_size, test_size and gap 
        k indicating the nr of folds, predefined to 10 
        n_best indicating the nr of features to select
        date_index, binary variable indicating if datetime will be used as spliting criteria or if it's the index of x
        
    Returns:
        n_best features
    
    """ """""" """"""
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.feature_selection import SequentialFeatureSelector
    from sklearn.model_selection import StratifiedKFold

    model = DecisionTreeClassifier(max_depth=6, random_state=0)

    skf = StratifiedKFold(
        n_splits=5,
    )

    forward_selection = SequentialFeatureSelector(
        model,
        direction="backward",
        cv=skf,
        n_features_to_select=n_best,
    )
    forward_selection = forward_selection.fit(x, y)

    subset = x.columns[forward_selection.get_support()]
    return (forward_selection, subset)


# embedded methods
def lasso_selection(x, y, coef_=0):
    """""" """""" """
    Performs lasso selection 
    Receives:
        x - pandas df containing input data  
        y - pandas df containing the target
        coef  - indicating the min absolute coeficient of the most important features, predefined to 0
        
    Returns series with coeficient of importance of each feature
    
    """ """""" """"""
    import time

    start_time = time.time()
    print("Performing Lasso Selection")

    from sklearn.linear_model import Lasso

    reg = Lasso(alpha=0.005, random_state=0)
    reg.fit(x, y)
    coef = pd.Series(reg.coef_, index=x.columns)

    print(
        "Lasso picked "
        + str(sum(coef != 0))
        + " variables and eliminated the other "
        + str(sum(coef == 0))
        + " variables"
    )
    print(f"Min absolute coef: {coef_}")
    coef = coef.loc[abs(coef.values) > coef_]

    print(f"Total variables in subset: {len(coef)}")
    print(" ")
    print("Lasso Selection lasted:")
    print("--- %s seconds ---" % (time.time() - start_time))
    return coef


def plot_importance(coef, name):
    imp_coef = coef.sort_values()
    plt.figure(figsize=(8, 10))
    imp_coef.plot(kind="barh")
    plt.title("Feature importance using " + name + " Model")
    plt.show()


# get feature importance
def get_FeatureImportance(name_list, feature_list):
    """""" """''
    it receives a list with the desired subset names and a list of lists containing the features of each subset
    The function creates a datafrane where each row represents a feature and the goal is to identify the most common features chosen by the different festure selection algorthms
    
    
    """ """"""
    t = pd.DataFrame(columns=x.columns, index=name_list)
    subset_list = feature_list

    for i, ii in zip(subset_list, np.arange(0, len(t.index))):
        t.iloc[ii][i] = 1

    t = t.T.fillna(0)

    t["totalCount"] = t.sum(axis=1)

    t = t.sort_values(by="totalCount", ascending=False)
    return t


# frop columns that are very correlated to each other
def dropCorrelatedVariables(data, subset, thresh):
    """""" """""
    Receives:
        pandas dataframe
        list with column names from the dataset - it can be a subset of columns or all columns
        thresh maximum value of absolute correlation allowed
    Calculates correlation matrix and creates a subset of features without absolute correlation above a certain threshold
    
    """ """""" ""
    corr = data[subset].corr()
    new_subset = []
    for col in subset:
        corr_temp = abs(corr[col]).sort_values(ascending=False).iloc[1:]
        corr_temp = corr_temp[corr_temp > 0.85]
        correlated_variables = corr_temp.index
        if len(corr_temp) == 0:
            new_subset.append(col)

        if len(corr_temp) > 0:
            count_ = 0
            for feature in correlated_variables:
                if feature in new_subset:
                    count_ = count_ + 1
            if count_ == 0:
                new_subset.append(col)

    return new_subset
