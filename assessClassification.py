import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def precision_recall_curve(model, names, X_train, y_train, x_val, y_val):
    """""" """""" """'
    Compares different models using stratified k fold for classification problems
    Receives:
        model and respective name (string)
        x_train (independent variables) and y_train (dependent variable)
        x_test (independent variables) and y_test (dependent variable)   
        
    Displays Precision - Recall  curve of the respective models 
    Note: the model is fitted inside the function 
    
    """ """""" """"""
    names = names
    model.fit(X_train, y_train)
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


def fitted_precision_recall_curve(model, names, X_train, y_train, x_val, y_val):
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


def roc_curve(models, names, X_train, y_train, X_val, y_val):
    """""" """""" """'
    Compares different models using stratified k fold for classification problems
    Receives:
        list with model names and a list of models 
        x_train (independent variables) and y_train (dependent variable)
        x_test (independent variables) and y_test (dependent variable)   
        
    Displays ROC curve of the respective models 
    
    """ """""" """"""

    names = names
    from sklearn.metrics import roc_curve

    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(10, 10))
    for (
        i,
        names,
    ) in zip(models, names):
        # i.fit(X_train,y_train)
        thresh_dict = {}
        prob_model = i.predict_proba(X_val)

        fpr_DT_ent, tpr_DT_ent, thresholds_DT_ent = roc_curve(y_val, prob_model[:, 1])
        plt.plot(
            fpr_DT_ent,
            tpr_DT_ent,
            label=f"ROC Curve {names}",
        )

        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.legend()
    plt.show()


def lift(test, pred, cardinaility=10):
    """""" """
    Receives:
        test - True Values of Target
        pred - predicted probability 
        cardinality - how granular we want to be 
    """ """"""

    res = pd.DataFrame(
        np.column_stack((test, pred)), columns=["Target", "PR_0", "PR_1"]
    )

    res["scr_grp"] = pd.qcut(res["PR_0"], cardinaility, labels=False) + 1

    crt = pd.crosstab(res.scr_grp, res.Target).reset_index()
    crt = crt.rename(columns={"Target": "Np", 0.0: "Negatives", 1.0: "Positives"})

    G = crt["Positives"].sum()
    B = crt["Negatives"].sum()

    avg_resp_rate = G / (G + B)

    crt["resp_rate"] = round(
        crt["Positives"] / (crt["Positives"] + crt["Negatives"]), 2
    )
    crt["lift"] = round((crt["resp_rate"] / avg_resp_rate), 2)
    crt["rand_resp"] = 1 / cardinaility
    crt["cmltv_p"] = round((crt["Positives"]).cumsum(), 2)
    crt["cmltv_p_perc"] = round(((crt["Positives"] / G).cumsum()) * 100, 1)
    crt["cmltv_n"] = round((crt["Negatives"]).cumsum(), 2)
    crt["cmltv_n_perc"] = round(((crt["Negatives"] / B).cumsum()) * 100, 1)
    crt["cmltv_rand_p_perc"] = (crt.rand_resp.cumsum()) * 100
    crt["cmltv_resp_rate"] = round(
        crt["cmltv_p"] / (crt["cmltv_p"] + crt["cmltv_n"]), 2
    )
    crt["cmltv_lift"] = round(crt["cmltv_resp_rate"] / avg_resp_rate, 2)
    crt["KS"] = round(crt["cmltv_p_perc"] - crt["cmltv_rand_p_perc"], 2)
    crt = crt.drop(
        [
            "rand_resp",
            "cmltv_p",
            "cmltv_n",
        ],
        axis=1,
    )

    print("average response rate: ", avg_resp_rate)
    return crt


# pred_withThreshold(rf_grid1, x_train_stand_out[subset4],0.32)
def pred_withThreshold(model, data, threshold):
    """""" """""" """
    Generates predictions for classification problems based on a certain threshold
    
    Receives
        model to generate predictions
        data to predict (usually test data)
        the threshold (eg if threshold = 0.7 then only observations with a probability greater than 0.7 will be 1)
        
    Returns: 
        list with predictions
    
    """ """""" """"""
    y_pred = ((model.predict_proba(data)[:, 1]) > threshold).astype(bool)
    return y_pred
