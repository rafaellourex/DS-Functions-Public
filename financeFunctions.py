import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

sns.set_style("whitegrid")


def get_target(
    df_engineer,
    t=1,
    upper_lower_multipliers=[2, 2],
    target_type="barrier",
    multi_t=None,
    price_type="smooth",
):
    """""" """"
    Receives:
        df_engineer - historical data sorted by ticker and date 
        t - nr of lookahead days 
        upper_lower_multipliers - list of size 2 containing the upper and lower multiplier of the Three Barrier Method
        target_type - type of target to be generated ['naive' or 'barrel']
            if naive then it simply uses future returns to generate the target
            if barriel then it uses the triple barrier method
        
    Creates targets for the Triple Barrier Method
    
    Returns:
        dicts where each key represents a different ticker
    """ """""" ""

    df_engineer = df_engineer.sort_values(["ticker", "date"])

    if target_type == "barrier":
        # set barrier variables
        upper = upper_lower_multipliers[0]
        lower = upper_lower_multipliers[1]
        companies = df_engineer.index.get_level_values(0).unique()

        # if we only have 1 t
        if multi_t == None:
            # generates label for 3 barrier strategy
            barriers = gen_BarrierLabel(
                df=df_engineer,
                companies=companies,
                up_tresh=upper,
                down_tresh=lower,
                lag=20,
                n=t,
            ).rename(columns={"out": f"Target_{t}"})

            train = pd.merge(
                df_engineer.reset_index(),
                barriers.reset_index()[["Target", "date", "ticker"]],
                on=["ticker", "date"],
            )
            train = train.set_index(["ticker", "date"]).dropna()
            # get x and y
            y = pd.DataFrame(index=train.index, data=np.sign(train["Target"]))
            y.loc[y["Target"] == 0, "Target"] = 1
            y = y.astype(int)
            x = train.drop(columns="Target")
            company = df_engineer.index.get_level_values(0).unique()

        if multi_t != None:
            for t in multi_t:
                # get barriers
                barriers = gen_BarrierLabel(
                    df=df_engineer,
                    companies=companies,
                    up_tresh=upper,
                    down_tresh=lower,
                    lag=20,
                    n=t,
                ).rename(columns={"out": f"Target_{t}"})

                # filter only important columns
                barriers = barriers.reset_index()[[f"Target_{t}", "date", "ticker"]]

                # get index
                index = barriers.set_index(["ticker", "date"]).index

                # filter y dataset to get index only present in the barriers dataset
                y = y.loc[y.index.isin(index)]

                # merge to get new Target column - at each iteration a new target colum is created based on t
                y = pd.merge(
                    left=y.reset_index(),
                    right=barriers[["ticker", "date", f"Target_{t}"]],
                    on=["ticker", "date"],
                    how="left",
                ).set_index(["ticker", "date"])

                # normalize targets
                # if Target score > 0.2 then consider 1
                y.loc[y[f"Target_{t}"] >= 0.2, f"Target_{t}"] = 1

                # if Target score < -0.2 then consider -1
                y.loc[y[f"Target_{t}"] <= -0.2, f"Target_{t}"] = -1

                # if Target score > -0.2 & score <0.2 then consider 0
                y.loc[
                    (y[f"Target_{t}"] > -0.2) & (y[f"Target_{t}"] < 0.2), f"Target_{t}"
                ] = 0

            y = (
                y.dropna()
                .drop_duplicates(["ticker", "date"])
                .set_index(["ticker", "date"])
            )
            x = x.loc[x.index.isin(y.index)]
            company = df_engineer.index.get_level_values(0).unique()

    if target_type == "naive":
        if multi_t == None:
            train = df_engineer.copy()
            fut_price = df_engineer.groupby("ticker")["close"].shift(-t)
            price = df_engineer["close"]
            target = (fut_price - price) / price
            target_bin = np.sign(target)
            train["fut_rerturns"] = target
            train["Target"] = np.sign(target)
            train.loc[train["Target"] == -1, "Target"] = 0

            x = train.iloc[:, :-2]
            y = train.iloc[:, -2:]
            y = y.dropna()
            x = x.loc[x.index.isin(y.index)]
            company = df_engineer.index.get_level_values(0).unique()

        if multi_t != None:
            y = pd.DataFrame(index=df_engineer.index)
            df_engineer["Target_MA"] = (
                df_engineer["close"].groupby("ticker").ewm(3).mean().values
            )
            for t in multi_t:
                # create a copy of close column
                if price_type == "smooth":
                    # calculate target based on EMA
                    fut_price = df_engineer["Target_MA"].groupby("ticker").shift(-t)
                    price = df_engineer["Target_MA"]

                if price_type != "smooth":
                    # calculate target based on normal closing price
                    # change
                    close = df_engineer["close"]
                    fut_price = close.groupby("ticker").shift(-t)
                    price = close

                fut_returns = (fut_price - price) / price
                target = np.sign(fut_returns)
                target = target.replace(0, -1)
                y[f"Target_{t}"] = target

            x = df_engineer.copy()
            y = y.dropna()
            x = x.loc[x.index.isin(y.index)]
            company = df_engineer.index.get_level_values(0).unique()

    # create dict for each coin
    data_x = {}
    data_y = {}
    for i in company:
        data_x[f"{i}"] = x.loc[x.index.get_level_values(0) == i]
        data_y[f"{i}"] = y.loc[y.index.get_level_values(0) == i]

    return (data_x, data_y)


def backtest(
    assess_df,
    init_value=10000,
    close="close",
    predictions="Predictions",
    date_="date",
    period=252,
):
    if date_ not in assess_df.columns:
        assess_df = assess_df.reset_index()

    mon_value = init_value
    hold = False
    data_dict = {}
    data_list = []

    for id_ in np.arange(0, len(assess_df)):
        data_dict = {}
        temp_data = assess_df.iloc[id_]
        date = temp_data[date_]
        pred = temp_data[predictions]
        price = temp_data[close]

        # if this happens then we buy
        if (pred == 1) and (hold == False):
            hold = True
            position = "Buy"
            nr_shares = mon_value / price
            shareValue = price * nr_shares
            mon_value = 0
            totValue = mon_value + shareValue

            data_dict["date"] = date
            data_dict["Price"] = price
            data_dict["Prediction"] = pred
            data_dict["Position"] = position
            data_dict["nrShares"] = nr_shares
            data_dict["sharesValue"] = shareValue
            data_dict["mon_value"] = mon_value
            data_dict["tot_value"] = totValue

        # if this happens then we hold
        elif (pred == 1) and (hold == True):
            position = "Hold"
            nr_shares = nr_shares
            shareValue = price * nr_shares
            mon_value = mon_value
            totValue = mon_value + shareValue

            data_dict["date"] = date
            data_dict["Price"] = price
            data_dict["Prediction"] = pred
            data_dict["Position"] = position
            data_dict["nrShares"] = nr_shares
            data_dict["sharesValue"] = shareValue
            data_dict["mon_value"] = mon_value
            data_dict["tot_value"] = totValue

        # if this happens then we sell
        elif pred == -1 and hold == True:
            hold = False
            position = "Sell"
            mon_value = nr_shares * price
            shareValue = 0
            nr_shares = 0
            totValue = mon_value + shareValue

            data_dict["date"] = date
            data_dict["Price"] = price
            data_dict["Prediction"] = pred
            data_dict["Position"] = position
            data_dict["nrShares"] = nr_shares
            data_dict["sharesValue"] = shareValue
            data_dict["mon_value"] = mon_value
            data_dict["tot_value"] = totValue

        # if this happens then we hold - we dont have any asset in this scenario
        elif pred == -1 and hold == False:
            position = "Hold"
            nr_shares = 0
            shareValue = price * nr_shares
            mon_value = mon_value
            totValue = mon_value + shareValue

            data_dict["date"] = date
            data_dict["Price"] = price
            data_dict["Prediction"] = pred
            data_dict["Position"] = position
            data_dict["nrShares"] = nr_shares
            data_dict["sharesValue"] = shareValue
            data_dict["mon_value"] = mon_value
            data_dict["tot_value"] = totValue

        data_list.append(data_dict)

    df_backtest = pd.DataFrame(data_list)
    df_backtest["strategyDailyReturns"] = df_backtest["tot_value"].pct_change()
    df_backtest["strategyCumulativeReturns"] = (
        df_backtest.tot_value / df_backtest.tot_value.iloc[0]
    )

    df_backtest["holdCumulativeReturns"] = df_backtest.Price / df_backtest.Price.iloc[0]
    df_backtest["holdValue"] = init_value * df_backtest["holdCumulativeReturns"]
    df_backtest["holdDailyReturns"] = df_backtest["holdValue"].pct_change()

    df_backtest.rename(columns={"tot_value": "strategyValue"}, inplace=True)

    holdSharpe = (df_backtest["strategyDailyReturns"].dropna().mean() * period) / (
        df_backtest["strategyDailyReturns"].dropna().std() * np.sqrt(period)
    )

    strategySharpe = (df_backtest["strategyDailyReturns"].dropna().mean() * period) / (
        df_backtest["strategyDailyReturns"].dropna().std() * np.sqrt(period)
    )

    assessStrategy_dict = {
        "sharpe": strategySharpe,
        "returns": (df_backtest["strategyDailyReturns"].dropna().mean() * period),
        "volatility": (
            df_backtest["strategyDailyReturns"].dropna().std() * np.sqrt(period)
        ),
    }

    title = "Personalized vs Hold Strategy - Returns"
    plt.figure(figsize=(15, 7))
    to_plot = (
        df_backtest.set_index(["date"])[
            ["strategyCumulativeReturns", "holdCumulativeReturns"]
        ]
        - 1
    )
    plt.plot(to_plot)
    plt.title(title, size=15)
    plt.ylabel(
        "Returns, in %",
    )
    plt.xlabel(
        "Date",
    )
    plt.legend(["Personalized Strategy", "Holdout Strategy"])
    # plt.show()

    title = f"Personalized vs Hold Strategy - End of Investment Monetary Value - Initial Investment: {init_value}"
    plt.figure(figsize=(15, 7))
    to_plot = df_backtest.set_index(["date"])[["strategyValue", "holdValue"]]
    plt.plot(to_plot)
    plt.title(title, size=15)
    plt.ylabel(
        "Returns, in %",
    )
    plt.xlabel(
        "Date",
    )
    plt.legend(
        ["Personalized Strategy - Total Value", "Holdout Strategy - Total Value"]
    )
    # plt.show()

    return (df_backtest, assessStrategy_dict)


"""
Calculates portfolio performance metrics sucha as sharpe, sortino Ratio, returns, volatility and max draxdown 

Receives:
    pandas series or DataFrame with multiple columns containing return

returns
    pandas df containing metric data
"""


def performReturnsAnslysis(df, periods=252):
    import quantstats.stats as qs

    sharpe = qs.sharpe(df, periods=periods, annualize=True)
    sortino = qs.sortino(df, periods=periods, annualize=True)

    returns = qs.avg_return(df, aggregate="year")
    volatility = qs.volatility(df, periods=periods, annualize=True)
    maxDraxDown = qs.max_drawdown(df)

    metricDict = {
        "returns": returns,
        "volatility": volatility,
        "sharpeRatio": sharpe,
        "sortinoRatio": sortino,
        "maxDraxdown": maxDraxDown,
    }
    metricsData = pd.DataFrame()
    for key in metricDict.keys():
        metricsData[key] = metricDict[key]

    return metricsData


# backtest_2(assess_df,10000)
