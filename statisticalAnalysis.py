import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns


def assessOLS(results, y):
    pvalues = results.pvalues.sort_values().plot.bar(figsize=(7, 5))
    plt.axhline(y=0.05, color="r", linestyle="--", linewidth=2)
    plt.axhline(y=0.1, color="g", linestyle="--", linewidth=2)
    title = "P-Values"
    plt.title(title, size=15, weight="bold")
    plt.ylabel("P-Value", weight="bold")
    plt.xlabel("Features", weight="bold")
    plt.tight_layout()
    plt.show()

    assess_ = pd.DataFrame(index=y.index)
    assess_["Predicted"] = results.resid.values
    assess_["True"] = y.values

    plt.figure(figsize=(7, 7))
    sns.regplot(
        data=assess_,
        x="True",
        y="Predicted",
    )
    title = "Predicted vs Real values"
    plt.title(title, size=15, weight="bold")
    plt.ylabel("Predicted", weight="bold")
    plt.xlabel("True", weight="bold")
    plt.tight_layout()
    plt.show()

    params = results.params.sort_values()
    params.plot.barh(figsize=(7, 10))
    title = "Coeficients"
    plt.title(title, size=15, weight="bold")
    plt.xlabel("Coeficients", weight="bold")
    plt.ylabel("Features", weight="bold")
    plt.tight_layout()
    plt.show()

    params = results.params
    stdError = results.bse
    to_plot = abs(
        pd.concat([params, stdError], axis=1, levels=["coef", "stdError"]).rename(
            columns={0: "coef", 1: "stdError"}
        )
    ).sort_values("coef")
    to_plot.plot.barh(figsize=(7, 10))
    title = "Absolute coeficients and standard error"
    plt.title(title, size=15, weight="bold")
    plt.xlabel("Coeficients", weight="bold")
    plt.ylabel("Features", weight="bold")
    plt.tight_layout()
    plt.show()


def performOLS(x, y, intercept=True, assess=True):
    import statsmodels.api as sm

    if intercept == True:
        print("OLS performed with intercept")
        x = sm.add_constant(x)

    ols_model = sm.OLS(y, x)
    results = ols_model.fit()

    pvalues = results.pvalues.sort_values()
    sig_fts = pvalues[pvalues.values <= 0.1].index
    print(results.summary())
    print(" ")
    print("=" * 100)
    print("Significant features at a 10% level:")
    print(pvalues[pvalues.values <= 0.1])
    print(sig_fts)
    if assess == True:
        assessOLS(results, y)
    return (ols_model, results)
