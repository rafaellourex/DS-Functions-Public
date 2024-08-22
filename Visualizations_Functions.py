import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("whitegrid")


def histogram(data):
    from math import ceil
    import pandas as pd
    import numpy as np

    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("whitegrid")
    print("Plotting Histograms")
    metric_features = (
        data.select_dtypes(include=np.number).set_index(data.index).columns
    )
    print(f"Nr of metrics features: {len(metric_features)}")
    n = 10

    if len(metric_features) > 20:
        for idx in np.arange(0, len(metric_features), n):
            f_idx = idx + n
            to_plot = data[metric_features].iloc[:, idx:f_idx]
            features = to_plot.columns

            fig, axes = plt.subplots(
                2, ceil(len(metric_features) / 2), figsize=(20, 11)
            )

            for ax, feat in zip(axes.flatten(), features):
                sns.histplot(to_plot[feat], ax=ax, color="teal")

                ax.set_title(feat)
                ax.set_xlabel("")
                ax.set_ylabel("")

                # plt.title(feat)

    if len(metric_features) <= 20:
        to_plot = data[metric_features]
        features = to_plot.columns
        fig, axes = plt.subplots(2, ceil(len(metric_features) / 2), figsize=(20, 11))

        for ax, feat in zip(axes.flatten(), metric_features):
            sns.histplot(to_plot[feat], ax=ax, color="teal")

            ax.set_title(feat)
            ax.set_xlabel("")
            ax.set_ylabel("")

    title = "Numeric Variables' Histograms"

    plt.suptitle(title)

    plt.show()
    return ()


def Histogram(data):
    metric_features = (
        data.select_dtypes(include=np.number).set_index(data.index).columns
    )

    print(f"Number of metric features : {len(metric_features)}")
    if len(metric_features) > 20:
        n = 10
        for idx in np.arange(0, len(metric_features), n):
            f_idx = idx + n
            to_plot = data[metric_features].iloc[:, idx:f_idx]
            histogram(to_plot)

    if len(metric_features) <= 20:
        histogram(data[metric_features])


def target_histogram(df, target):
    from math import ceil
    import pandas as pd
    import numpy as np
    import seaborn as sns

    metric_features = df.select_dtypes(include=np.number).columns
    # All Numeric Variables' Histograms in one figure

    # Prepare figure. Create individual axes where each histogram will be placed
    fig, axes = plt.subplots(5, ceil(len(metric_features) / 5), figsize=(20, 20))

    # Plot data
    # Iterate across axes objects and associate each histogram:
    for ax, feat in zip(axes.flatten(), metric_features):
        sns.histplot(
            x=df[feat],
            hue=df[f"{target}"],
            stat="density",
            common_norm=False,
            kde=True,
            element="step",
            color="blue",
            linewidth=2,
            ax=ax,
        )
        ax.set_title(feat)
        ax.set_xlabel("")
        ax.set_ylabel("")

    title = "Numeric Variables' Histograms by Target"
    plt.suptitle(title)
    plt.subplots_adjust(wspace=0.2, hspace=0.5)

    plt.show()


def Target_Histogram(data, target):
    metric_features = (
        data.select_dtypes(include=np.number).set_index(data.index).columns
    )

    print(f"Number of metric features : {len(metric_features)}")
    if len(metric_features) > 20:
        print("Plotting Multiple Figures")
        n = 10
        for idx in np.arange(0, len(metric_features), n):
            f_idx = idx + n
            to_plot = data[metric_features].iloc[:, idx:f_idx]
            new_subset = []
            for col in to_plot.columns:
                if col != target:
                    new_subset.append(col)

            to_plot = pd.concat([to_plot[new_subset], data[target]], axis=1)
            target_histogram(to_plot, target)

    if len(metric_features) <= 20:
        new_subset = []
        print("Plotting a Sinle Figure")
        for col in metric_features:
            if col != target:
                new_subset.append(col)

        to_plot = pd.concat([data[new_subset], data[target]], axis=1)
        target_histogram(to_plot, target)


def boxplot(data):
    from math import ceil

    # All Numeric Variables' Histograms in one figure
    metric_features = (
        data.select_dtypes(include=np.number).set_index(data.index).columns
    )
    sns.set_style("whitegrid")
    # Prepare figure. Create individual axes where each histogram will be placed
    fig, axes = plt.subplots(4, ceil(len(metric_features) / 4), figsize=(30, 15))

    # Plot data
    # Iterate across axes objects and associate each histogram (hint: use the ax.hist() instead of plt.hist()):
    for ax, feat in zip(
        axes.flatten(), metric_features
    ):  # Notice the zip() function and flatten() method

        sns.boxplot(x=data[feat].values, ax=ax)
        ax.set_title(feat)
        ax.set_xlabel("")
        ax.set_ylabel("")

    # Layout
    # Add a centered title to the figure:
    title = "Numeric Variables' Histograms"

    plt.suptitle(title)

    plt.show()


def Boxplot(data):
    metric_features = (
        data.select_dtypes(include=np.number).set_index(data.index).columns
    )

    print(f"Number of metric features : {len(metric_features)}")
    if len(metric_features) > 20:
        n = 10
        for idx in np.arange(0, len(metric_features), n):
            f_idx = idx + n
            to_plot = data[metric_features].iloc[:, idx:f_idx]
            boxplot(to_plot)

    if len(metric_features) <= 20:
        boxplot(data[metric_features])


def target_boxplot(data, target):
    from math import ceil

    metric_features = data.select_dtypes(include=np.number).columns
    # data = pd.concat([data[metric_features], data[target]])
    # All Numeric Variables' Box Plots in one figure

    # Prepare figure. Create individual axes where each box plot will be placed
    fig, axes = plt.subplots(3, ceil(len(metric_features) / 3), figsize=(20, 11))
    print(data.columns)
    # Plot data
    # Iterate across axes objects and associate each box plot:
    for ax, feat in zip(axes.flatten(), metric_features):
        sns.boxplot(x=data[target], y=data[feat], ax=ax, palette=["turquoise", "teal"])
        ax.set_title(f"{feat}")
        ax.set_ylabel("")
        ax.set_xlabel("")

    # Layout
    # Add a centered title to the figure:
    title = "Numeric Variables' Box Plots by Target"
    plt.suptitle(title)
    plt.subplots_adjust(wspace=0.3)

    plt.show()


def Target_Boxplot(data, target):
    import pandas as pd

    metric_features = data.select_dtypes(include=np.number).columns
    print(f"Number of metric features : {len(metric_features)}")
    if len(metric_features) > 20:
        print("Plotting Multiple Figures")
        n = 10
        for idx in np.arange(0, len(metric_features), n):
            f_idx = idx + n
            to_plot = data[metric_features].iloc[:, idx:f_idx]
            new_subset = []
            for col in to_plot.columns:
                if col != target:
                    new_subset.append(col)

            to_plot = pd.concat([to_plot[new_subset], data[target]], axis=1)
            target_boxplot(to_plot, target)

    if len(metric_features) <= 20:
        new_subset = []
        print("Plotting a Sinle Figure")
        for col in metric_features:
            if col != target:
                new_subset.append(col)

        to_plot = pd.concat([data[new_subset], data[target]], axis=1)
        target_boxplot(to_plot, target)


def univariate_target_analysis(data, target):
    for i in data.select_dtypes(exclude=np.object0).columns[1:]:
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))
        sns.violinplot(
            ax=axes[0], x=data[target], y=data[i], palette=["turquoise", "teal"]
        )
        sns.distplot(data[i], hist=True, ax=axes[1], color="teal")
        sns.boxplot(
            ax=axes[2], x=data[target], y=data[i], palette=["turquoise", "teal"]
        )
    plt.show()


def barplot(data, cat, num):
    """""" """
    Receives
        pandas dataframe 
        cat - name of a category column
        num - name of a numerical column
    
    """ """"""
    # groupby taking into account the category
    data = data.groupby(f"{cat}").mean()[f"{num}"].sort_values(ascending=False)

    # set title and size
    title = f"Average {num} by {cat}"
    size = 15

    # plot barploy
    plt.figure(figsize=(10, 7))
    sns.barplot(
        x=data.index,
        y=data,
        color="royalblue",
    )

    # set title
    plt.title(title)
    # set xlabel
    plt.xlabel(cat, size=size)
    # set ylabel
    plt.ylabel(num, size=size)
    # set xticks
    plt.xticks(
        rotation=45,
    )

    plt.show()


def target_categorical_analysis(df, target):
    from math import ceil

    non_metric_features = (
        df.drop(columns=[target]).select_dtypes(exclude=np.number).columns
    )
    fig, axes = plt.subplots(3, ceil(len(non_metric_features) / 3), figsize=(21, 17))

    # Plot data
    # Iterate across axes objects and associate each bar plot:
    for ax, feat in zip(axes.flatten(), non_metric_features):
        data = (
            df.groupby([target])[feat]
            .value_counts(normalize=True)
            .rename("prop")
            .reset_index()
        )
        sns.barplot(
            data=data,
            x=feat,
            y="prop",
            hue=f"{target}",
            ax=ax,
            palette=["turquoise", "teal"],
        )
        ax.set_title(feat)
        ax.set_xlabel("")

    title = "Categorical Variables' Relative Frequencies by Target"
    plt.suptitle(title)
    # Rotating X-axis target
    axes.flatten()[0].tick_params(axis="x", labelrotation=90)
    axes.flatten()[2].tick_params(axis="x", labelrotation=90)
    axes.flatten()[-1].remove()
    plt.subplots_adjust(wspace=0.3, hspace=0.7)

    plt.show()


def scatter(data):
    from math import ceil

    metric_features = list(data.select_dtypes(include=np.number).columns)
    fig, axes = plt.subplots(2, ceil(len(metric_features) / 2), figsize=(20, 10))
    for i in metric_features:
        col = np.array(metric_features)
        col = col[col != i]
        for ii, ax in zip(col, axes.flatten()):
            # plt.figure(index)
            sns.scatterplot(
                data=data,
                x=data[i],
                y=data[ii],
                ax=ax,
            )
    title = "ScatterPlot Hue"
    plt.suptitle(title, weight="bold")
    plt.show()


def scatter_hue(data, hue):
    from math import ceil

    metric_features = list(data.select_dtypes(include=np.number).columns)
    fig, axes = plt.subplots(2, ceil(len(metric_features) / 2), figsize=(20, 10))
    for i in metric_features:
        col = np.array(metric_features)
        col = col[col != i]
        for ii, ax in zip(col, axes.flatten()):
            # plt.figure(index)
            sns.scatterplot(data=data, x=data[i], y=data[ii], hue=data[hue], ax=ax)
    title = "ScatterPlot Hue"
    plt.suptitle(title, weight="bold")
    plt.show()


def categorical_relation(data, cat_cat_var1, cat_var2):
    """""" """"
    Receives
        pandas datagrame 
        cat_cat_var1 - category 1 variable we want to plot
        cat_var2 - category 2 variable we want to plot
        
    Plots the relatiionship between categories by plotting how many times each category in category 1 is in each category of category 2
    """ """""" ""
    df_counts = (
        data.groupby([f"{cat_cat_var1}", f"{cat_var2}"])
        .size()
        .unstack()
        .plot.bar(stacked=True, figsize=(10, 7))
    )
    plt.title(f"{cat_cat_var1} vs {cat_var2} - relation")
    plt.xticks(rotation=45, weight="bold")


def categorical_TargetProfile(data, target):
    """""" """"
    Receives:
        pandas dataframe 
        name of target feature 
        
    Performs Target profile by plotting boxplots and histograms 
    Note: more usefull when we have 

    """ """""" ""
    print("Ploting Boxplots")
    Target_Boxplot(data, target)

    print("Ploting Histograms")
    Target_Histogram(data, target)


def leverage_analysis(data, value_col):
    """""" """"
    Receives:
        pandas df 
        value_col - column that will be used to calculate value generated by each observation *eg: it can be total amount spend by a specific customer)
    
    Performs
        lleverage analysis considering a specific value_column
    """ """""" ""

    # sort values by value, highest to lowest
    data = data.sort_values(ascending=False, by=f"{value_col}")
    # create a value pct_column,
    # respective value divided by total value
    data["value_pct"] = data[f"{value_col}"] / np.sum(data[f"{value_col}"])
    # create cumulative sum of the value,
    # the last observation will make it 1
    data["value_cumsum"] = data["value_pct"].cumsum()

    # create % of population that 1 observstion represents
    data["id_pct"] = 1 / len(data)
    # create cumulative sum
    data["id_cumsum"] = data["id_pct"].cumsum()

    cust_sig = data.copy()
    cust_leverage = pd.DataFrame(index=["HighValue", "MediumValue", "LowValue"])
    high_value = cust_sig.loc[cust_sig["id_cumsum"] <= 0.3, f"{value_col}"].sum()
    medium_value = cust_sig.loc[
        (cust_sig["id_cumsum"] > 0.3) & (cust_sig["id_cumsum"] <= 0.5), f"{value_col}"
    ].sum()
    low_value = cust_sig.loc[(cust_sig["id_cumsum"] > 0.5), f"{value_col}"].sum()
    values = [high_value, medium_value, low_value]
    cust_leverage["Value"] = values
    cust_leverage["Pop_pct"] = [0.3, 0.2, 0.5]
    total_value = cust_sig.loc[:, f"{value_col}"].sum()
    cust_leverage["Value_pct"] = cust_leverage["Value"] / total_value
    cust_leverage["Leverage"] = cust_leverage["Value_pct"] / cust_leverage["Pop_pct"]

    # plot leverage plot
    plt.figure(figsize=(10, 7))
    axis = sns.lineplot(y=data["value_cumsum"], x=data["id_cumsum"], color="Turquoise")
    axis.set_ylabel(f"% {value_col} cuulative value")
    axis.set_xlabel("% of population")
    plt.title("Product Leverage")
    return (data, cust_leverage)


def cor_heat_map(corr):
    """""" """""" """""" """
    Receives 
        correlation matrix in pandas df format 
        
    Plots Correlation matrix
    
    """ """""" """""" """"""
    plt.figure(figsize=(30, 30))
    sns.heatmap(
        corr,
        vmax=1,
        vmin=-1,
        linewidths=0.1,
        annot=True,
        annot_kws={"size": 10},
        square=True,
        cmap=sns.diverging_palette(220, 10, as_cmap=True),
    )
    plt.show()


def cor_heat_map_half(corr, titie="Correlation Matrix"):
    """""" """""" """""" """
    Receives 
        correlation matrix in pandas df format 
        
    Plots Correlation matrix
    
    """ """""" """""" """"""

    corr = corr
    mask = np.triu(np.ones_like(corr, dtype=bool))

    plt.figure(figsize=(30, 30))
    sns.heatmap(
        corr,
        vmax=1,
        vmin=-1,
        linewidths=0.1,
        annot=True,
        annot_kws={"size": 10},
        square=True,
        cmap=sns.diverging_palette(220, 10, as_cmap=True),
        mask=mask,
    )
    plt.title(titie, size=15)
    plt.tight_layout()
    plt.show()


def Heatmap(data, title="Heatmap", min_value=0, max_value=10):

    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 8))
    # Set the min and max values for the palette
    min_value = min_value
    max_value = max_value

    # Set the color palette
    cmap = sns.diverging_palette(10, 150, s=80, l=55, n=9, center="light")

    # Plot the heatmap
    sns.heatmap(data, cmap=cmap, vmin=min_value, vmax=max_value, square=True)

    plt.title(title, fontsize=16, fontweight="bold", pad=20)

    # Show the plot
    plt.show()


def cluster_barplot_profile(data, target):
    from math import ceil

    metric_features = data.select_dtypes(include=np.number).columns

    # All Numeric Variables' Box Plots in one figure

    # Prepare figure. Create individual axes where each box plot will be placed
    fig, axes = plt.subplots(2, ceil(len(metric_features) / 2), figsize=(30, 30))

    # Plot data
    # Iterate across axes objects and associate each box plot:
    for ax, feat, i in zip(axes.flatten(), metric_features, range(len(axes.flatten()))):

        if feat != target:
            sns.barplot(
                x=data.groupby(f"{target}").mean()[feat].index,
                y=data.groupby(f"{target}").mean()[feat].values,
                color="royalblue",
                ax=ax,
            )
            ax.axhline(y=data[feat].mean(), color="red")
            ax.tick_params(labelrotation=45)
            ax.title.set_text(f"{feat}")

    title = "Barplot Cludtering Profile"

    plt.suptitle(title)

    plt.show()
