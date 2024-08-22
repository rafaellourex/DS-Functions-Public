import numpy as np
import pandas as pd

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


# get skf - cross validation methodology
def get_skf(x, y, cv_type, train_size=10, test_size=1, gap=0, date_column="date"):
    """
    A function to return different types of cross-validation objects
    based on the input cv_type.
    """
    # import libraries
    from sklearn.model_selection import KFold

    if cv_type == "kfold":
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
    else:
        print(
            "skf_type provided is not valid. Please try: 'kfold' , TimeCV_block",
            "TimeCV_continuous",
        )

    return skf


# fit scaler - normalization method
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


# fit cross validation and store metrics
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
        scores - pd dataframe containing the scores
    """
    # import libraries
    import pandas as pd
    import numpy as np
    from sklearn.metrics import (
        mean_absolute_percentage_error,
        mean_squared_error,
        r2_score,
        mean_absolute_error,
    )

    scores_list = []
    splits_list = []
    Splits_dict = dict()
    predictions_df = pd.DataFrame()

    if scaler == True:
        print("Scalling will be performed at each iteration")
        print(f"Scaler type: {scaler_type}")

    if over_sampling == True:
        print("SMOTE oversampling will be performed")

    if under_sampling == True:

        print("Undersampling will be performed:")
        print("Strategy: Random Undersampling")

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
            MAPE_train = mean_absolute_percentage_error(Y_train, pred_train)
            r2_train = r2_score(Y_train, pred_train)
            mse_train = mean_squared_error(Y_train, pred_train)
            rmse_train = np.sqrt(mean_squared_error(Y_train, pred_train))
            mae_train = mean_absolute_error(Y_train, pred_train)

            # calculate metrics for the test data
            MAPE_test = mean_absolute_percentage_error(Y_test, pred_test)
            r2_test = r2_score(Y_test, pred_test)
            mse_test = mean_squared_error(Y_test, pred_test)
            rmse_test = np.sqrt(mean_squared_error(Y_test, pred_test))
            mae_test = mean_absolute_error(Y_test, pred_test)

            # create a temp_dict to store values of the respective split
            scores_dict = dict()
            scores_dict["r2_train"] = r2_train
            scores_dict["rmse_train"] = rmse_train
            scores_dict["mse_train"] = mse_train
            scores_dict["mape_train"] = MAPE_train
            scores_dict["mae_train"] = mae_train

            scores_dict["r2_test"] = r2_test
            scores_dict["rmse_test"] = rmse_test
            scores_dict["mse_test"] = mse_test
            scores_dict["mape_test"] = MAPE_test
            scores_dict["mae_test"] = mae_test

            scores_list.append(scores_dict)

            if predictions == True:
                pred_dict = dict()
                pred_df = pd.DataFrame(index=Y_test.index)
                pred_df["Target"] = Y_test.values
                pred_df["Predictions"] = model.predict(x_test)
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
    cv_type="kfold",
    scaler_type="stand",
    train_size=10,
    test_size=1,
    gap=0,
    date_column="date",
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
    )

    return returns
