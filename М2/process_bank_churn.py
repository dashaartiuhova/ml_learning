# =========================
# Required Imports
# =========================
from typing import Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer

# =========================
# Data Splitting
# =========================
def split_data(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 12,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split dataset into train and validation sets.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    target_col : str
        Name of the target column.
    test_size : float, optional
        Validation set proportion.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    Tuple containing X_train, X_val, y_train, y_val
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    return train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )


# =========================
# Column Selection
# =========================
def get_column_types(
    df: pd.DataFrame,
) -> Tuple[pd.Index, pd.Index]:
    """
    Identify numeric and categorical columns.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    Tuple of (numeric_columns, categorical_columns)
    """
    numeric_cols = df.select_dtypes(include="number").columns
    categorical_cols = df.select_dtypes(include="object").columns
    return numeric_cols, categorical_cols

# =========================
# Scaling
# =========================
def scale_numeric_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    numeric_cols,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Scale numeric features using MinMaxScaler.
    """
    scaler = MinMaxScaler()

    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])

    return X_train, X_val, scaler


# =========================
# Encoding
# =========================
def encode_categorical_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    categorical_cols,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    One-hot encode categorical features.
    Unknown categories are ignored.
    """
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

    encoded_train = encoder.fit_transform(X_train[categorical_cols])
    encoded_val = encoder.transform(X_val[categorical_cols])

    encoded_cols = encoder.get_feature_names_out(categorical_cols)

    encoded_train_df = pd.DataFrame(
        encoded_train, columns=encoded_cols, index=X_train.index
    )
    encoded_val_df = pd.DataFrame(
        encoded_val, columns=encoded_cols, index=X_val.index
    )

    # Drop original categorical columns
    X_train = X_train.drop(columns=categorical_cols)
    X_val = X_val.drop(columns=categorical_cols)

    # Concatenate encoded features
    X_train = pd.concat([X_train, encoded_train_df], axis=1)
    X_val = pd.concat([X_val, encoded_val_df], axis=1)

    return X_train, X_val, encoder
from typing import Tuple, Optional
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

def preprocess_data(
    raw_data: pd.DataFrame,
    target_col: str = "Exited",
    scale_numeric: bool = True,
) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.Series, pd.Series,
    pd.Index, Optional[MinMaxScaler], Optional[OneHotEncoder]
]:
    """
    Full preprocessing pipeline:
    - Drop unnecessary columns
    - Split dataset
    - Optionally scale numeric features
    - Encode categorical variables

    Parameters
    ----------
    raw_data : pd.DataFrame
        Original dataset.
    target_col : str
        Target column name.
    scale_numeric : bool
        Whether to apply MinMax scaling to numeric features.

    Returns
    -------
    X_train : pd.DataFrame
        Processed training features.
    X_val : pd.DataFrame
        Processed validation features.
    y_train : pd.Series
        Training target.
    y_val : pd.Series
        Validation target.
    feature_cols : pd.Index
        List of feature columns after processing (numeric + categorical before encoding).
    scaler : MinMaxScaler or None
        Fitted scaler (if scale_numeric=True).
    encoder : OneHotEncoder
        Fitted encoder for categorical variables.
    """

    df = raw_data.copy()

    if "Surname" in df.columns:
        df = df.drop(columns=["Surname"])

    # Split
    X_train, X_val, y_train, y_val = split_data(df, target_col)

    # Column types
    numeric_cols, categorical_cols = get_column_types(X_train)

    # Scaling
    scaler = None
    if scale_numeric:
        X_train, X_val, scaler = scale_numeric_features(
            X_train, X_val, numeric_cols
        )

    # Encoding
    X_train, X_val, encoder = encode_categorical_features(
        X_train, X_val, categorical_cols
    )
    feature_names = numeric_cols.tolist() + categorical_cols.tolist()
    return X_train, X_val, y_train, y_val, feature_names, scaler, encoder


def preprocess_test_data(
    df_test: pd.DataFrame,
    feature_names,
    scaler=None,
    encoder=None,
):
    """
    Preprocess test set using fitted scaler and encoder from training data.
    """

    X_test = df_test.copy()

    # Drop columns that training used
    if "Surname" in X_test.columns:
        X_test = X_test.drop(columns=["Surname"])

    # Keep only the same features as training
    numeric_cols = [col for col in feature_names if col in X_test.select_dtypes(include="number").columns]
    categorical_cols = [col for col in feature_names if col in X_test.select_dtypes(include="object").columns]

    # Scaling numeric features
    if scaler is not None and numeric_cols:
        X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    # Encoding categorical features
    if encoder is not None and categorical_cols:
        encoded_test = encoder.transform(X_test[categorical_cols])
        encoded_cols = encoder.get_feature_names_out(categorical_cols)
        encoded_test_df = pd.DataFrame(encoded_test, columns=encoded_cols, index=X_test.index)
        X_test = X_test.drop(columns=categorical_cols)
        X_test = pd.concat([X_test, encoded_test_df], axis=1)

    return X_test
