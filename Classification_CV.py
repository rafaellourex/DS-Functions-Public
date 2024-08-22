from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from dateutil.relativedelta import *


class TimeBasedCV_Continuous(object):
    """
    Parameters
    ----------
    train_period: int
        number of time units to include in each train set
        default is 30
    test_period: int
        number of time units to include in each test set
        default is 7
    freq: string
        frequency of input parameters. possible values are: days, months, years, weeks, hours, minutes, seconds
        possible values designed to be used by dateutil.relativedelta class
        deafault is days

    the training size increases at each split (same beggining different ends)
    """

    def __init__(self, train_period=30, test_period=7, freq="days"):
        self.train_period = train_period
        self.test_period = test_period
        self.freq = freq

    def split(self, data, validation_split_date=None, date_column="record_date", gap=0):
        """
        Generate indices to split data into training and test set

        Parameters
        ----------
        data: pandas DataFrame
            your data, contain one column for the record date
        validation_split_date: datetime.date()
            first date to perform the splitting on.
            if not provided will set to be the minimum date in the data after the first training set
        date_column: string, deafult='record_date'
            date of each record
        gap: int, default=0
            for cases the test set does not come right after the train set,
            *gap* days are left between train and test sets

        Returns
        -------
        train_index ,test_index:
            list of tuples (train index, test index) similar to sklearn model selection
        """

        # check that date_column exist in the data:
        try:
            data[date_column]
        except:
            raise KeyError(date_column)

        train_indices_list = []
        test_indices_list = []
        split_dates = pd.DataFrame()
        train_start = []
        train_end = []
        test_start = []
        test_end = []
        a = 1
        if validation_split_date == None:
            validation_split_date = data[date_column].min().date() + eval(
                "relativedelta(" + self.freq + "=self.train_period)"
            )

        start_train = validation_split_date - eval(
            "relativedelta(" + self.freq + "=self.train_period)"
        )
        end_train = start_train + eval(
            "relativedelta(" + self.freq + "=self.train_period)"
        )
        start_test = end_train + eval("relativedelta(" + self.freq + "=gap)")
        end_test = start_test + eval(
            "relativedelta(" + self.freq + "=self.test_period)"
        )

        a = 0
        while end_test <= data[date_column].max().date():
            a = a + 30

            # train indices:
            cur_train_indices = list(
                data[
                    (data[date_column].dt.date >= start_train)
                    & (data[date_column].dt.date < end_train)
                ].index
            )

            # test indices:
            cur_test_indices = list(
                data[
                    (data[date_column].dt.date >= start_test)
                    & (data[date_column].dt.date < end_test)
                ].index
            )

            #             print("Train period:",start_train,"-" , end_train, ", Test period", start_test, "-", end_test,
            #                   "# train records", len(cur_train_indices), ", # test records", len(cur_test_indices))

            train_indices_list.append(cur_train_indices)
            test_indices_list.append(cur_test_indices)
            train_start.append(start_train)
            test_start.append(start_test)
            train_end.append(end_train)
            test_end.append(end_test)
            # update dates:
            start_train = validation_split_date
            end_train = start_train + eval(
                "relativedelta(" + self.freq + f"={self.train_period+a})"
            )

            start_test = end_train + eval("relativedelta(" + self.freq + "=gap)")
            end_test = start_test + eval(
                "relativedelta(" + self.freq + "=self.test_period)"
            )

        # mimic sklearn output
        index_output = [
            (train, test) for train, test in zip(train_indices_list, test_indices_list)
        ]

        self.n_splits = len(index_output)
        split_dates["Start_Train"] = train_start
        split_dates["End_Train"] = train_end
        split_dates["Start_Test"] = test_start
        split_dates["End_Test"] = test_end

        return (index_output, split_dates)

    def get_n_splits(self):
        """Returns the number of splitting iterations in the cross-validator
        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        return (self.n_splits, split_dates)


class TimeBasedCV_block(object):
    """
    Parameters
    ----------
    train_period: int
        number of time units to include in each train set
        default is 30
    test_period: int
        number of time units to include in each test set
        default is 7
    freq: string
        frequency of input parameters. possible values are: days, months, years, weeks, hours, minutes, seconds
        possible values designed to be used by dateutil.relativedelta class
        deafault is days

    the training splits have always the same size and move forward to the future at each split
    (different begginings, different ends)
    """

    def __init__(self, train_period=30, test_period=7, freq="days"):
        self.train_period = train_period
        self.test_period = test_period
        self.freq = freq

    def split(self, data, validation_split_date=None, date_column="record_date", gap=0):
        """
        Generate indices to split data into training and test set

        Parameters
        ----------
        data: pandas DataFrame
            your data, contain one column for the record date
        validation_split_date: datetime.date()
            first date to perform the splitting on.
            if not provided will set to be the minimum date in the data after the first training set
        date_column: string, deafult='record_date'
            date of each record
        gap: int, default=0
            for cases the test set does not come right after the train set,
            *gap* days are left between train and test sets

        Returns
        -------
        train_index ,test_index:
            list of tuples (train index, test index) similar to sklearn model selection
        """

        # check that date_column exist in the data:
        try:
            data[date_column]
        except:
            raise KeyError(date_column)

        train_indices_list = []
        test_indices_list = []
        split_dates = pd.DataFrame()
        train_start = []
        train_end = []
        test_start = []
        test_end = []

        if validation_split_date == None:
            validation_split_date = data[date_column].min().date() + eval(
                "relativedelta(" + self.freq + "=self.train_period)"
            )

        start_train = validation_split_date - eval(
            "relativedelta(" + self.freq + "=self.train_period)"
        )
        end_train = start_train + eval(
            "relativedelta(" + self.freq + "=self.train_period)"
        )
        start_test = end_train + eval("relativedelta(" + self.freq + "=gap)")
        end_test = start_test + eval(
            "relativedelta(" + self.freq + "=self.test_period)"
        )

        while end_test <= data[date_column].max().date():
            # train indices:
            cur_train_indices = list(
                data[
                    (data[date_column].dt.date >= start_train)
                    & (data[date_column].dt.date < end_train)
                ].index
            )

            # test indices:
            cur_test_indices = list(
                data[
                    (data[date_column].dt.date >= start_test)
                    & (data[date_column].dt.date < end_test)
                ].index
            )

            #             print("Train period:",start_train,"-" , end_train, ", Test period", start_test, "-", end_test,
            #                   "# train records", len(cur_train_indices), ", # test records", len(cur_test_indices))

            train_indices_list.append(cur_train_indices)
            test_indices_list.append(cur_test_indices)
            train_start.append(start_train)
            test_start.append(start_test)
            train_end.append(end_train)
            test_end.append(end_test)
            # update dates:
            start_train = start_train + eval(
                "relativedelta(" + self.freq + "=self.test_period)"
            )
            end_train = start_train + eval(
                "relativedelta(" + self.freq + "=self.train_period)"
            )
            start_test = end_train + eval("relativedelta(" + self.freq + "=gap)")
            end_test = start_test + eval(
                "relativedelta(" + self.freq + "=self.test_period)"
            )

        # mimic sklearn output
        index_output = [
            (train, test) for train, test in zip(train_indices_list, test_indices_list)
        ]

        self.n_splits = len(index_output)
        split_dates["Start_Train"] = train_start
        split_dates["End_Train"] = train_end
        split_dates["Start_Test"] = test_start
        split_dates["End_Test"] = test_end

        return (index_output, split_dates)

    def get_n_splits(self):
        """Returns the number of splitting iterations in the cross-validator
        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        return (self.n_splits, split_dates)


def TS_SkipCV_index(x, y, train_lenght, test_size, skip):
    """""" """""" """''
    Generates the indexes to perform sliding window CV for time series where the trainig period is the same throughout the iterations 
    The function enables cross validation with different train and test sizes.
    Receives:
        x_train (independent variables) and y_train (dependent variable)
        train lengtth (nr of days that will be used for training)
        test lengtth (nr of days that will be used for testing)
        
    Note: train and test lenght are calculated using indexation (eg: if train length = 30, then 30 observations will be used for taining )
        
    Returns:
        indexes of each split
    
    """ """""" """""" ""

    split_dict = dict()

    for index in np.arange(0, len(x) - test_size - skip - 2):

        if index == 0:
            start_index = index
        else:
            start_index = start_index + test_size

        final_index = start_index + train_lenght
        test_index = final_index + (skip)

        x_test = x.iloc[test_index : test_index + test_size]
        if test_index + test_size <= len(x):
            if len(x_test) > 0:
                train_index = np.arange(start_index, final_index, 1)
                test_index = np.arange(test_index, test_index + test_size, 1)
                indexes = [train_index, test_index]
                split_dict[f"Split_{index}"] = indexes

    return split_dict


def holdoutSplit_stratify(x, y):
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, shuffle=True, stratify=y
    )

    return (X_train, X_test, y_train, y_test)


def holdoutSplit(x, y):
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, shuffle=True
    )

    return (X_train, X_test, y_train, y_test)


def get_skf(x, y, cv_type, train_size=10, test_size=1, gap=0, date_column="date"):
    """
    A function to return different types of cross-validation objects based on the input cv_type.

    Parameters:
        x : pandas DataFrame
            Independent features.
        y : pandas DataFrame or Series
            Target data.
        cv_type : str
            Type of cross-validation to be performed.
        train_size : int, optional
            Number of splits for training data. Default is 10.
        test_size : int, optional
            Number of splits for test data. Default is 1.
        gap : int, optional
            Gap between train and test sets for time-based cross-validation. Default is 0.
        date_column : str, optional
            Name of the column containing date information. Default is 'date'.

    Returns:
        skf : generator object
            Cross-validation iterator.
    """
    # import libraries
    from sklearn.model_selection import StratifiedKFold, KFold

    if cv_type == "stratKfold":
        print(cv_type)
        print(f"Nr of splits: {train_size}")
        skf = StratifiedKFold(n_splits=train_size, shuffle=True, random_state=0).split(
            x, y
        )

    elif cv_type == "kfold":
        print(cv_type)
        print(f"Nr of splits: {train_size}")
        skf = KFold(n_splits=train_size, shuffle=True, random_state=0).split(x, y)

    elif cv_type == "TimeCV_block":
        print(cv_type)
        print(f"Nr of splits: {train_size}")
        X = x.reset_index()
        X["date"] = pd.to_datetime(X["date"])

        tscv = TimeBasedCV_block(
            train_period=train_size, test_period=test_size, freq="days"
        )
        skf, split_df = tscv.split(X, date_column=date_column)

    elif cv_type == "TimeCV_continuous":
        print(cv_type)
        print(f"Nr of splits: {train_size}")
        X = x.reset_index()
        X["date"] = pd.to_datetime(X["date"])

        tscv = TimeBasedCV_block(
            train_period=train_size, test_period=test_size, freq="days"
        )
        skf, split_df = tscv.split(X, date_column=date_column)

    elif cv_type == "WK_CV":
        print(cv_type)
        print(f"Train size: {train_size}")
        print(f"Test size: {test_size}")
        print(f"Gap: {gap}")
        skf = TS_SkipCV_index(
            x, y, train_lenght=train_size, test_size=test_size, skip=gap
        ).values()

    elif cv_type == "WK_Block":
        print(cv_type)
        print(f"Train size: {train_size}")
        print(f"Test size: {test_size}")
        print(f"Gap: {gap}")
        from sklearn.model_selection import StratifiedKFold

        skf = TimeBasedCV_block(
            train_period=train_size, test_period=test_size, freq="days"
        ).split(x.reset_index(), date_column="date", gap=gap)[0]

    elif cv_type == "WK_cont":
        print(cv_type)
        print(f"Train size: {train_size}")
        print(f"Test size: {test_size}")
        print(f"Gap: {gap}")
        tscv = TimeBasedCV_Continuous(
            train_period=train_size, test_period=test_size, freq="days"
        )
        skf, split_df = tscv.split(x.reset_index(), date_column="date", gap=gap)

    else:
        print(
            "skf_type provided is not valid. Please try: 'stratKfold' or 'kfold' or 'WK_CV' or 'TimeCV_block' or 'TimeCV_continuous' "
        )

    return skf


def fit_scaler(x_train, x_val, scaler_type="stand"):
    """
    Function to fit a scaler to the training data and transform the training and validation sets.
    Parameters:
        x_train (pd.DataFrame): Training data
        x_val (pd.DataFrame): Validation data
        scaler_type (str): Type of scaler to use ('stand', 'minmax', 'robust')
    Returns:
        train_scaled (pd.DataFrame): Scaled training data
        val_scaled (pd.DataFrame): Scaled validation data
        scaler: fitted scaler
    """
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

    x = pd.DataFrame(x_train)
    numeric_features = x.select_dtypes(include=np.number).columns
    non_numeric_features = x.select_dtypes(exclude=np.number).columns

    scaler_types = {
        "stand": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler(),
    }

    scaler = scaler_types.get(scaler_type)
    scaler.fit(x[numeric_features])
    x_train_scaled = scaler.transform(x[numeric_features])
    x_train_scaled = pd.DataFrame(
        x_train_scaled, columns=numeric_features, index=x.index
    )

    x_val_scaled = pd.DataFrame(
        scaler.transform(x_val[numeric_features]),
        index=x_val.index,
        columns=numeric_features,
    )
    train_scaled = pd.concat([x_train_scaled, x_train[non_numeric_features]], axis=1)
    val_scaled = pd.concat([x_val_scaled, x_val[non_numeric_features]], axis=1)

    return train_scaled, val_scaled, scaler


def fit_CV(
    X,
    Y,
    model,
    skf,
    scaler=False,
    over_sampling=False,
    under_sampling=False,
    predictions=False,
    scaler_type="stand",
    average="binary",
):
    """
    Receives:
        X - input data for the model
        Y - model target
        model - model being used
        skf - list containing the splits
        scaler - if true normalization will be performed at each split (typeof normalization depends on scaler_type) - the model used for normalization will be returned
        over_sampling - if true smote will be applied to the training data at each split (the est data remains the same to avoid missleading results)
        predictions - if true predictions are returned
        scaler_type - type of normalization to be performed (stand; minmax; robust)
    Returns:
        scores - dictionary containing relevant information about the model performance and the trained model itself
                 keys  = ['scores', 'splits','model','predictions' (opt),'scaler' (opt)]

    """
    # import libraries
    import pandas as pd
    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        f1_score,
        recall_score,
        roc_auc_score,
    )

    scores_list = []
    splits_list = []
    Splits_dict = dict()
    predictions_df = pd.DataFrame()

    if scaler == True:
        print("Scalling will be performed at each iteration")
        print(f"Scaler type: {scaler_type}")
        print(" ")

    if over_sampling == True:
        print("SMOTE oversampling will be performed")
        print(" ")

    if under_sampling == True:

        print("Undersampling will be performed:")
        print("Strategy: Random Undersampling")
        print(" ")

    print(f"Methodology to calculate metrics: {average}")
    print(" ")

    # for each split
    index_ = 0
    for train_index, test_index in skf:
        try:
            # get data corresponding to each split
            x_train, x_test = X.iloc[train_index], X.iloc[test_index]
            Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

            if scaler == True:
                # run fit_scaler function
                x_train, x_test, scaler_model = fit_scaler(
                    x_train=x_train, x_val=x_test, scaler_type=scaler_type
                )

            if over_sampling == True:
                from imblearn.over_sampling import SMOTE

                # initiate SMOTE
                os = SMOTE(random_state=0, k_neighbors=5)
                # fit SMOTE only to the training data
                x_train, Y_train = os.fit_resample(x_train, Y_train)

            if under_sampling == True:
                import imblearn.under_sampling as underSample

                us = underSample.RandomUnderSampler(random_state=0)
                x_train, Y_train = us.fit_resample(x_train, Y_train)

            # fit model
            model = model.fit(x_train, Y_train)

            splits_dict = dict()
            splits_dict["y_train"] = Y_train
            splits_dict["y_test"] = Y_test
            Splits_dict[f"split_{index_}"] = splits_dict
            index_ = index_ + 1
            splits_list.append(splits_dict)

            pred_train = model.predict(x_train)
            pred_test = model.predict(x_test)

            # calculate metrics for the training data
            f1_train = f1_score(Y_train, pred_train, average=average)
            recall_train = recall_score(Y_train, pred_train, average=average)
            precision_train = precision_score(Y_train, pred_train, average=average)
            auc_train = roc_auc_score(Y_train, pred_train, average=average)
            accuracy_train = accuracy_score(Y_train, pred_train)

            # calculate metrics for the test data
            f1_test = f1_score(Y_test, pred_test, average=average)
            recall_test = recall_score(Y_test, pred_test, average=average)
            precision_test = precision_score(Y_test, pred_test, average=average)
            auc_test = roc_auc_score(Y_test, pred_test, average=average)
            accuracy_test = accuracy_score(Y_test, pred_test)

            # create a temp_dict to store values of the respective split
            scores_dict = dict()
            scores_dict["accuracy_train"] = accuracy_train
            scores_dict["f1_train"] = f1_train
            scores_dict["recall_train"] = recall_train
            scores_dict["precision_train"] = precision_train
            scores_dict["auc_train"] = auc_train
            scores_dict["accuracy_test"] = accuracy_test
            scores_dict["f1_test"] = f1_test
            scores_dict["recall_test"] = recall_test
            scores_dict["precision_test"] = precision_test
            scores_dict["auc_test"] = auc_test
            scores_list.append(scores_dict)

            if predictions == True:
                pred_dict = dict()
                pred_df = pd.DataFrame(index=Y_test.index)
                pred_df["Target"] = Y_test.values
                pred_df["Predictions"] = model.predict(x_test)
                pred_df["Probability"] = model.predict_proba(x_test)[:, 1]
                predictions_df = pd.concat([predictions_df, pred_df], axis=0)
        except Exception as error:
            print(error)
    scores = pd.DataFrame.from_records(scores_list)
    print(scores.mean())
    returns = dict()
    returns["scores"] = scores
    returns["splits"] = Splits_dict
    returns["model"] = model

    if predictions == True:
        returns["predictions"] = predictions_df
    if scaler == True:
        returns["scaler"] = scaler_model

    return returns


# wrapper function to connect all the functions above


def validateModel(
    X_train,
    y_train,
    model,
    scaler=False,
    over_sampling=False,
    under_sampling=False,
    predictions=False,
    cv_type="stratKfold",
    scaler_type="stand",
    train_size=10,
    test_size=1,
    gap=0,
    date_column="date",
    average="binary",
):
    """""" """""" """'
    Receives:
        x_train (independent variables) and 
        y_train (dependent variable)
        model to perform validation
        scaler - binary feature in case we want to standardize 
        over_sampling - binary in case we want to apply smote
        
    Dependencies:
        fit_CV
        
    Returns:
        model 
        scores      
    
    """ """""" """""" ""
    from sklearn.metrics import f1_score, recall_score, precision_score

    skf = get_skf(
        X_train,
        y_train,
        cv_type=cv_type,
        train_size=train_size,
        test_size=test_size,
        gap=gap,
        date_column=date_column,
    )

    returns = fit_CV(
        X=X_train,
        Y=y_train,
        model=model,
        skf=skf,
        scaler=scaler,
        over_sampling=over_sampling,
        under_sampling=under_sampling,
        scaler_type=scaler_type,
        predictions=predictions,
        average=average,
    )

    return returns


def metrics(y_train, pred_train, y_val, pred_val):
    """""" """""" """'
    Performs a classification metric report 
    Receives:
        model and respective name (string)
        x_train (independent variables) and y_train (dependent variable)
        x_test (independent variables) and y_test (dependent variable)   
        
    Displays Precision - Recall  curve of the respective models 
    Note: the model is fitted inside the function 
    
    """ """""" """"""
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix

    print(
        "___________________________________________________________________________________________________________"
    )
    print(
        "                                                     TRAIN                                                 "
    )
    print(
        "-----------------------------------------------------------------------------------------------------------"
    )
    print(classification_report(y_train, pred_train))
    print(confusion_matrix(y_train, pred_train))

    print(
        "___________________________________________________________________________________________________________"
    )
    print(
        "                                                VALIDATION                                                 "
    )
    print(
        "-----------------------------------------------------------------------------------------------------------"
    )
    print(classification_report(y_val, pred_val))
    print(confusion_matrix(y_val, pred_val))


def grid_search(
    model,
    params,
    x_train,
    y_train,
    x_val,
    y_val,
    metric,
    cv_type="stratKfold",
    n_splts=10,
    refit=True,
):
    """""" """""" """
   performs gridsearch for a given model
    
    Receives:
        model and a dict with parameters to be optimied
        x_train (independent variables) and y_train (dependent variable)
        x_test (independent variables) and y_test (dependent variable)
        
    Note: it's designed for classification problems but can be adjusted for regression
    
    Requirements:
        metrics
        
    Returns:
        fitted gridsearch object
      
    Available Metrics for classification: 
    'accuracy'
    'f1_macro',
    'f1_micro',
    'f1_samples',
    'f1_weighted'
    'precision',
    'precision_macro',
    'precision_micro',
    'precision_samples',
    'precision_weighted'
    'recall',
    'recall_macro',
    'recall_micro',
    'recall_samples',
    'recall_weighted',
    'roc_auc',
    'roc_auc_ovo',
    'roc_auc_ovo_weighted',
    'roc_auc_ovr',
    """ """""" """"""
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import f1_score

    print("Performing Grid Search")
    print(f"Perfomance metric {metric}")
    print(f"Optimizing: {model}")
    print(f"Cross Validation: {cv_type}")

    grid = GridSearchCV(
        model, param_grid=params, cv=n_splts, scoring=metric, refit=refit
    ).fit(x_train, y_train)
    best_est = grid.best_estimator_
    grid_pred = best_est.predict(x_val)
    metrics(y_train, best_est.predict(x_train), y_val, grid_pred)
    print(" ")
    print("Best Estimator:")
    print(best_est)
    return grid


def fitted_precision_recall_curve(model, names, x_val, y_val):
    """""" """""" """'
    Compares different models using stratified k fold for classification problems
    Receives:
        model and respective name (string)
        x_train (independent variables) and y_train (dependent variable)
        x_test (independent variables) and y_test (dependent variable)   
        
    Displays Precision - Recall  curve of the respective models 
    Note: the model is fitted outside the function 
    
    """ """""" """"""
    names = names
    # model.fit(X_train,y_train)
    proba = model.predict_proba(x_val)
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(y_val, proba[:, 1])

    # apply f1 score
    fscore = (2 * precision * recall) / (precision + recall)
    # locate the index of the largest f score
    ix = np.argmax(fscore)
    print("Best Threshold=%f, F-Score=%.3f" % (thresholds[ix], fscore[ix]))

    plt.plot(recall, precision, marker=".", label=f"{names}")
    plt.scatter(recall[ix], precision[ix], marker="o", color="black", label="Best")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.show()


def assessModelPerformance(target, prediction, model_name=None, average="binary"):
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
    )

    # Calculate accuracy
    accuracy = accuracy_score(target, prediction)

    # Calculate precision
    precision = precision_score(target, prediction, average=average)

    # Calculate recall
    recall = recall_score(target, prediction, average=average)

    # Calculate F1 score
    f1 = f1_score(target, prediction, average=average)

    # Calculate AUC-ROC score
    auc_roc = roc_auc_score(target, prediction, average=average)

    # Create a dictionary to store the performance metrics
    performance_metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "AUC-ROC Score": auc_roc,
    }

    if model_name is not None:
        performance_metrics["Model"] = model_name

    return performance_metrics


def assessModel(model, x_test, y_test, model_name=None, average="binary"):
    pred = model.predict(x_test)
    prob = model.predict_proba(x_test)[:, :]
    assess_df = y_test.reset_index().copy()
    assess_df["pred"] = pred
    assess_df["pred_prob"] = prob[:, 1]

    performance_metrics = assessModelPerformance(
        y_test, pred, model_name=model_name, average=average
    )

    fitted_precision_recall_curve(
        model, names="Precision Recall Curve", x_val=x_test, y_val=y_test
    )
    return (assess_df, performance_metrics)


def ValidateModel(
    x_train,
    y_train,
    x_test,
    y_test,
    model,
    model_name=None,
    scaler=False,
    over_sampling=False,
    under_sampling=False,
    predictions=False,
    cv_type="stratKfold",
    scaler_type="stand",
    train_size=10,
    test_size=1,
    gap=0,
    date_column="date",
    average="binary",
):

    assess_cv = validateModel(
        X_train=x_train,
        y_train=y_train,
        model=model,
        scaler=scaler,
        over_sampling=over_sampling,
        under_sampling=under_sampling,
        predictions=predictions,
        cv_type=cv_type,
        scaler_type=scaler_type,
        train_size=train_size,
        test_size=test_size,
        gap=gap,
        date_column=date_column,
        average=average,
    )

    if scaler == True:
        scaler_model = assess_cv["scaler"]
        x_test = pd.DataFrame(
            data=scaler_model.transform(x_test),
            index=x_test.index,
            columns=x_test.columns,
        )
    model_trained = assess_cv["model"]
    assess_df, performance_metrics = assessModel(
        model=model_trained,
        x_test=x_test,
        y_test=y_test,
        model_name=model_name,
        average=average,
    )
    assess_cv["predictions_test"] = assess_df
    assess_cv["scores_test"] = performance_metrics
    return assess_cv


def calculate_metrics(y_test, y_pred):
    """
    Calculate accuracy, F1 score, precision, and recall metrics.

    Parameters:
        y_test (array-like): True labels.
        y_pred (array-like): Predicted labels.

    Returns:
        dict: A dictionary containing calculated metrics.
    """
    metrics = {}

    # Calculate accuracy
    metrics["Accuracy"] = accuracy_score(y_test, y_pred)

    # Calculate F1 score
    metrics["F1"] = f1_score(y_test, y_pred)

    # Calculate precision
    metrics["Precision"] = precision_score(y_test, y_pred)

    # Calculate recall
    metrics["Recall"] = recall_score(y_test, y_pred)

    metrics["Auc"] = roc_auc_score(y_test, y_pred)

    return metrics


def metrics_test(y_val, pred_val):
    """""" """""" """'
    Performs a classification metric report 
    Receives:
        model and respective name (string)
        x_train (independent variables) and y_train (dependent variable)
        x_test (independent variables) and y_test (dependent variable)   
        
    Displays Precision - Recall  curve of the respective models 
    Note: the model is fitted inside the function 
    
    """ """""" """"""
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix

    print(
        "___________________________________________________________________________________________________________"
    )
    print(
        "                                                VALIDATION                                                 "
    )
    print(
        "-----------------------------------------------------------------------------------------------------------"
    )
    print(classification_report(y_val, pred_val))
    print(confusion_matrix(y_val, pred_val))

    metrics = calculate_metrics(y_val, pred_val)
    return metrics


def trainModels(
    df_x_train,
    y_train,
    df_x_test,
    y_test,
    subset,
    model_dict,
    scaler=True,
    cv_type="stratKfold",
):
    """
    Trains multiple models, validates them using cross-validation, and evaluates on test data.

    Receives:
        df_x_train : pandas DataFrame
            Training dataset with independent features.
        y_train : pandas DataFrame or Series
            Target data for training.
        df_x_test : pandas DataFrame
            Test dataset with independent features.
        y_test : pandas DataFrame or Series
            Target data for testing
        subset : list
            List of column names to be used as independent features.
        model_dict : dict
            Dictionary where each key represents a model and the value the respective model
            (e.g., model_dict = {'LogisticRegression': LogisticRegression(random_state=0)}).
        scaler : bool, optional
            Flag indicating whether scaling should be used in training. Default is True.
        cv_type : str, optional
            Type of cross-validation to be performed during training. Default is 'stratKfold'.

    Returns:
        tuple
            A tuple containing:
            - df_assess_final_lasso : pandas DataFrame
                DataFrame containing validation and test scores for each model.
            - assess_dict : dict
                Dictionary containing detailed assessment results for each model.

    """

    # Initialize DataFrame to store cross-validation scores.
    full_scores_rfe = pd.DataFrame()
    # Initialize dictionary to store detailed assessment results.
    assess_dict = {}
    # Initialize list to store test metrics for each model.
    test_metrics_list = []

    # Iterate over each model in the model dictionary.
    for model_name in model_dict.keys():
        # Validate the model using cross-validation.
        assess = validateModel(
            model=model_dict[model_name],
            X_train=df_x_train[subset],
            y_train=y_train,
            scaler=scaler,
            cv_type=cv_type,
        )

        # Extract validation scores.
        scores = assess["scores"]
        scores_test = scores.iloc[:, :]
        scores_test["model"] = model_name

        # Append validation scores to the DataFrame.
        full_scores_rfe = pd.concat([full_scores_rfe, scores_test], axis=0)

        # Create a dictionary to store detailed assessment results.
        assess_d = dict()
        assess_d["model"] = assess["model"]
        assess_d["scores"] = scores_test
        assess_d["scaler"] = assess["scaler"]

        # Store assessment results in the assessment dictionary.
        assess_dict[model_name] = assess_d

        # Get the trained model from the assessment results.
        model = assess["model"]

        # Transform test data if scaling is enabled.
        if scaler == True:
            scaler_model = assess["scaler"]
            x_test_transform = scaler_model.transform(df_x_test[subset])
        else:
            x_test_transform = df_x_test[subset]

        # Make predictions on the test data.
        pred_test = model.predict(x_test_transform)

        # Calculate test metrics.
        test_metrics = metrics_test(y_test, pred_test)
        test_metrics["model"] = model_name

        # Append test metrics to the list.
        test_metrics_list.append(test_metrics)

    # Calculate mean validation scores for each model.
    scores_grouped = full_scores_rfe.groupby("model").mean()
    scores_grouped = scores_grouped.iloc[:, -4:]

    # Rename columns to indicate they represent validation scores.
    for c in scores_grouped.columns:
        scores_grouped = scores_grouped.rename(columns={c: c.replace("test", "val")})

    # Create DataFrame for test metrics.
    assess_test_df = pd.DataFrame(test_metrics_list).set_index("model")

    # Rename columns to indicate they represent test scores.
    for c in assess_test_df.columns:
        assess_test_df = assess_test_df.rename(columns={c: f"{c.lower()}_test"})

    # Display cross-validation scores.
    print("Cross-Validation Scores:")
    display(scores_grouped)

    print(" ")
    # Display test scores.
    print("Test Scores:")
    display(assess_test_df)

    # Concatenate validation and test scores into a final DataFrame.
    df_assess_final_lasso = pd.concat([scores_grouped, assess_test_df], axis=1)

    return (df_assess_final_lasso, assess_dict)


def plotROC(X_test, y_test, results_dict):
    """
    Plots ROC curves for multiple models based on their prediction probabilities.

    Parameters:
    -----------
    X_test : pandas DataFrame
        Test dataset with independent features.
    y_test : pandas DataFrame or Series
        True labels for the test dataset.
    results_dict : dict
        Dictionary containing model results including trained models and scalers.

    Returns:
    --------
    None

    """

    # Import necessary libraries.
    from sklearn.metrics import roc_curve
    import matplotlib.pyplot as plt

    # Create a figure for the ROC curve.
    plt.figure(figsize=(10, 10))

    # Plot the random ROC curve.
    plt.plot([0, 1], [0, 1], "k--", label="Random")

    # Iterate over each model in the results dictionary.
    for i in results_dict.keys():
        # Extract the model and scaler from the results dictionary.
        model = results_dict[i]["model"]
        scaler = results_dict[i]["scaler"]

        # Standardize test data using the scaler.
        x_test_stand = scaler.transform(X_test)

        # Predict probabilities for positive class.
        y_pred_proba = model.predict_proba(x_test_stand)[:, 1]

        # Compute ROC curve.
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

        # Plot the ROC curve for the model.
        plt.plot(fpr, tpr, label=i)

    # Add labels and title to the plot.
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve", weight="bold", size=12)

    # Add legend to the plot.
    plt.legend(loc="lower right")

    # Show the plot.
    plt.show()


def plot_cm(X_test, y_test, results_dict):
    """
    Plot confusion matrix for each model in the results dictionary.

    Parameters:
        X_test : array-like of shape (n_samples, n_features)
            Test samples.
        y_test : array-like of shape (n_samples,)
            True labels for X_test.
        results_dict : dict
            Dictionary containing model results with 'model' and 'scaler' keys.

    """

    # Import required libraries.
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    # Iterate over each model in the results dictionary.
    for i in results_dict.keys():
        # Extract model and scaler from results dictionary.
        model = results_dict[i]["model"]
        scaler = results_dict[i]["scaler"]

        # Standardize test data using the scaler.
        x_test_stand = scaler.transform(X_test)

        # Predict labels using the model.
        y_pred = model.predict(x_test_stand)

        # Compute confusion matrix.
        matrix = confusion_matrix(y_test, y_pred)

        # Display confusion matrix.
        disp = ConfusionMatrixDisplay(
            confusion_matrix=matrix,
            display_labels=model.classes_,
        )

        # Plot confusion matrix.
        plt.figure(figsize=(10, 10))
        disp.plot()
        plt.title(f"{i} - Confusion Matrix", weight="bold", size=12)
        plt.show()


"""
Usage example 

"""
# rfe_subset = ['Year_Birth', 'MntGoldProds', 'total_accepted', 'is_alone',
#               'AcceptedCmp5', 'AcceptedCmp3', 'NumWebVisitsMonth',
#               'NumStorePurchases', 'NumDealsPurchases', 'loyalty', 'MntMeatProducts',
#               'MntFruits', 'Teenhome', 'MntFishProducts', 'Recency']
# subset = rfe_subset
# model_dict = instantiate_models()
# assess_rfe, assess_rfe_dict = trainModels(df_x_train=X_train,
#                                           y_train=y_train,
#                                           df_x_test=X_test,
#                                           y_test=y_test,
#                                           subset=subset,
#                                           model_dict=model_dict)


# plotROC(X_test[subset], y_test, assess_rfe_dict)
# plot_cm(X_test[subset], y_test, assess_rfe_dict)
