import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple
import joblib
from sklearn.metrics import fbeta_score

from imblearn.over_sampling import SMOTENC, SMOTEN
from imblearn.combine import SMOTETomek
from sklearn.decomposition import PCA
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    roc_curve,
    auc,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    r2_score,
    mean_squared_error,
)
from imblearn.datasets import fetch_datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_score, recall_score


pd.set_option("display.max.rows", 130)
pd.set_option("display.max.columns", 130)
pd.set_option("float_format", "{:.2f}".format)
scaler = StandardScaler()


def feature_engineering_light(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform feature engineering on the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.

    Returns
    -------
    pd.DataFrame
        Dataset with engineered features.
    """

    # Example: Create a new feature based on existing ones
    def month_to_season(month):
        if month in ["dec", "jan", "feb"]:
            return "winter"
        elif month in ["mar", "apr", "may"]:
            return "spring"
        elif month in ["jun", "jul", "aug"]:
            return "summer"
        else:  # sep, oct, nov
            return "autumn"

    df["season"] = df["month"].apply(month_to_season)
    df["education"] = df["education"].replace("illiterate", "unknown")

    bins = [0, 18, 35, 50, 65, np.inf]  
    labels = ['Child','Young Adult', 'Adult', 'Middle Age', 'Senior'] 

    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False,include_lowest=True)
    df.drop(columns=['age','month'],inplace=True)
    return df

def feature_engineering(df: pd.DataFrame, min_freq: float = 0.01) -> pd.DataFrame:
    """
    Feature engineering with filtered interactions to avoid high cardinality.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    min_freq : float
        Minimum relative frequency for interaction categories to be kept, others go to 'Other'.

    Returns
    -------
    pd.DataFrame
        Dataset with engineered features.
    """

    # Season from month
    def month_to_season(month):
        if month in ["dec", "jan", "feb"]:
            return "winter"
        elif month in ["mar", "apr", "may"]:
            return "spring"
        elif month in ["jun", "jul", "aug"]:
            return "summer"
        else:
            return "autumn"

    df["season"] = df["month"].apply(month_to_season)

    # Education cleanup
    df["education"] = df["education"].replace("illiterate", "unknown")

    # Age groups
    bins = [0, 18, 35, 50, 65, np.inf]
    labels = ['Child','Young Adult','Adult','Middle Age','Senior']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False, include_lowest=True)

    # Previous contact
    df['has_prev_contact'] = (df['pdays'] != 999).astype(int)
    df['previous_success'] = (df['poutcome'] == 'success').astype(int)

    # Avg success by category
    for col in ['job','marital','education','contact']:
        df[f'avg_success_by_{col}'] = df.groupby(col)['previous_success'].transform('mean')

    # Filtered interactions to reduce cardinality
    def filter_interaction(col1, col2, name, round_col=None):
        if round_col is not None:
            col2_series = df[col2].round(round_col).astype(str)
        else:
            col2_series = df[col2].astype(str)
        inter = df[col1].astype(str) + "_" + col2_series
        freq = inter.value_counts(normalize=True)
        valid = freq[freq >= min_freq].index
        df[name] = inter.where(inter.isin(valid), 'Other')

    filter_interaction('job','previous','job_prev_interaction')
    filter_interaction('marital','campaign','marital_campaign_interaction')
    filter_interaction('education', 'emp.var.rate', 'education_empvar_interaction', round_col=1)

    # Drop unused raw columns
    df.drop(columns=['age','month'], inplace=True)

    return df

def clean_data_from_outliers(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Remove outliers from a specified column using the IQR method.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    column : str
        Name of the column to clean.

    Returns
    -------
    pd.DataFrame
        Dataset with outliers removed from the specified column.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    print(f"Removing outliers from '{column}' using IQR method:")
    print(f"Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
    print(f"Lower bound: {lower_bound:.2f}, Upper bound: {upper_bound:.2f}")
    print(f"Original dataset shape: {df.shape}")
    print(
        f"Dataset shape after outlier removal: {df[(df[column] >= lower_bound) & (df[column] <= upper_bound)].shape}"
    )

    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

  
def encode_categorical_features(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    categorical_columns: list,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Encode categorical features using One-Hot Encoding.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    categorical_columns : list
        List of categorical column names to encode.

    Returns
    -------
    pd.DataFrame
        Dataset with encoded categorical features.
    """
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoder.fit(df_train[categorical_columns])
    encoded_cols = list(encoder.get_feature_names_out(categorical_columns))
    df_train[encoded_cols] = encoder.transform(df_train[categorical_columns])
    df_val[encoded_cols] = encoder.transform(df_val[categorical_columns])
    if df_test is not None:
        df_test[encoded_cols] = encoder.transform(df_test[categorical_columns])

    print('Drop:',categorical_columns)

    df_train.drop(columns=categorical_columns, inplace=True)
    df_val.drop(columns=categorical_columns, inplace=True)
    if df_test is not None:
        df_test.drop(columns=categorical_columns, inplace=True)    


    return df_train, df_val, df_test,encoded_cols


def scale_numerical_features(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    numerical_columns: list,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Scale numerical features using StandardScaler.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    numerical_columns : list
        List of numerical column names to scale.

    Returns
    -------
    pd.DataFrame
        Dataset with scaled numerical features.
    """
    scaler.fit(df_train[numerical_columns])
    df_train[numerical_columns] = scaler.transform(df_train[numerical_columns])
    df_val[numerical_columns] = scaler.transform(df_val[numerical_columns])
    df_test[numerical_columns] = scaler.transform(df_test[numerical_columns])

    return df_train, df_val, df_test

    
def find_best_threshold_f2(y_true, probas, beta=2):
    """
    Find the best threshold to maximize F-beta score.

    Parameters
    ----------
    y_true : array-like
        True labels (0/1).
    probas : array-like
        Predicted probabilities for positive class.
    beta : float
        Beta parameter for F-beta score (default 2).

    Returns
    -------
    best_t : float
        Best threshold.
    best_fbeta : float
        F-beta score at best threshold.
    best_recall : float
        Recall at best threshold.
    best_precision : float
        Precision at best threshold.
    """
    import numpy as np
    from sklearn.metrics import recall_score, precision_score

    thresholds = np.linspace(0.1, 0.9, 50)
    best_t = thresholds[0]
    best_fbeta = 0
    best_recall = 0
    best_precision = 0

    for t in thresholds:
        preds = (probas >= t).astype(int)
        fbeta = fbeta_score(y_true, preds, beta=beta)
        recall_t = recall_score(y_true, preds)
        precision_t = precision_score(y_true, preds)

        if fbeta > best_fbeta:
            best_fbeta = fbeta
            best_t = t
            best_recall = recall_t
            best_precision = precision_t

    return best_t, best_fbeta, best_recall, best_precision
