import pandas as pd
import numpy as np


def confusion_table(true_labels, pred_labels) -> pd.DataFrame:
    assert all(
        el in [0, 1] for el in np.unique(true_labels)
    ), "mis configured values in true labels"
    assert all(
        el in [0, 1] for el in np.unique(pred_labels)
    ), "mis configured values in predictions"

    check = np.array(
        [
            (
                int(el1 == 0 and el2 == 0),  # correct neg
                int(el1 == 0 and el2 == 1),  # incorrect neg
                int(el1 == 1 and el2 == 0),  # incorrect pos
                int(el1 == 1 and el2 == 1),
            )  # correct pos
            for el1, el2 in zip(pred_labels, true_labels)
        ]
    )

    mtrx = np.array([np.sum(check[:, i]) for i in range(4)]).reshape(2, 2)
    mtrx = np.vstack((mtrx, np.sum(mtrx, axis=0)))

    return pd.DataFrame(
        data=mtrx,
        columns=pd.MultiIndex.from_tuples([("y_true", 0), ("y_true", 1)]),
        index=pd.MultiIndex.from_tuples([("y_pred", 0), ("y_pred", 1), ("Total", "")]),
    )


def total_error_rate(confusion_table: pd.DataFrame) -> float:
    total = confusion_table.loc[("Total", "")].sum()
    return 1.0 - np.trace(confusion_table) / total


def true_positive_rate(confusion_table: pd.DataFrame) -> float:
    """
    The sensitivity, the % of actual Trues correctly classified.
    """
    return (
        confusion_table.loc[("y_pred", 1), ("y_true", 1)]
        / confusion_table.loc[("Total", "")].iloc[1]
    )


def false_positive_rate(confusion_table: pd.DataFrame) -> float:
    """
    The % of actual Trues that are mis classified
    """
    return (
        confusion_table.loc[("y_pred", 1), ("y_true", 1)]
        / confusion_table.loc[("Total", "")].iloc[1]
    )


def true_negative_rate(confusion_table: pd.DataFrame) -> float:
    """
    Specificity, the % of actual False's that are correctly identified.
    """
    return (
        confusion_table.loc[("y_pred", 1), ("y_true", 1)]
        / confusion_table.loc[("Total", "")].iloc[1]
    )


def some_func(x: float, df: pd.DataFrame) -> pd.DataFrame:
    """

    Parameters
    ----------
    x : _type_
        _description_
    df : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """


def nCp(sigma2, estimator, X, Y):
    "Negative Cp statistic"
    n, p = X.shape
    Yhat = estimator.predict(X)
    RSS = np.sum((Y - Yhat) ** 2)
    return -(RSS + 2 * p * sigma2) / n
