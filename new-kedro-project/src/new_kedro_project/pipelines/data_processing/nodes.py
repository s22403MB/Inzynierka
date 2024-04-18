import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import logging
from typing import Dict


#def p

def process_patients(df):
    grouped = df.groupby(['Participant', 'Category Right']).size().unstack(fill_value=0)

    df['total_fixation'] = df['Participant'].map(grouped['Fixation'])
    df['total_saccade'] = df['Participant'].map(grouped['Saccade'])
    df['total_blink'] = df['Participant'].map(grouped['Blink'])
    df = df[['Participant', 'y', 'total_fixation', 'total_saccade', 'total_blink']]
    df = df.drop_duplicates()
    logger = logging.getLogger(__name__)
    logger.info(df.head())
    # df['total_blink'] = df['total_blink'].apply(lambda x: np.log(x))
    # df['total_saccade'] = df['total_saccade'].apply(lambda x: np.log(x))
    scaleMinMax(df, 'total_fixation')
    scaleMinMax(df, 'total_saccade')
    scaleMinMax(df, 'total_blink')
    return df


def scaleMinMax(dataFrame, columnName):
    scaler = MinMaxScaler()
    series = dataFrame[columnName]
    scaled_data = scaler.fit_transform(series.values.reshape(-1,1))
    scaled_series = pd.Series(scaled_data.flatten(), index=series.index)
    dataFrame[columnName] = scaled_series


def split_patients(df, parameters: Dict):
    X = df[parameters["features"]]
    y = df['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=parameters["test_size"],
                                                        random_state=parameters["random_state"])
    return X_train, X_test, y_train, y_test


def patients_fit_model(X_train, y_train, parameters: Dict):
    model = RandomForestClassifier(random_state=parameters["random_state"])
    model.fit(X_train, y_train)
    return model


def rate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    # Ocena dokładności modelu
    accuracy = accuracy_score(y_test, y_pred)
    logger = logging.getLogger(__name__)
    logger.info("Dokładność modelu: {:.2f}%".format(accuracy * 100))

    # Raport klasyfikacji
    print(classification_report(y_test, y_pred))


def cross_val(model, X, y, parameters: Dict):
    cv_scores = cross_val_score(model, X, y, cv=parameters["k_fold"])
    logger = logging.getLogger(__name__)
    logger.info(f"Wyniki {parameters['k_fold']}-krotnej walidacji krzyżowej dla modelu RandomForestClassifier:")
    logger.info(cv_scores)
    logger.info(f"Średnia dokładność: {np.mean(cv_scores):.2f} ± {np.std(cv_scores):.2f}")


def _is_true(x: pd.Series) -> pd.Series:
    return x == "t"


def _parse_percentage(x: pd.Series) -> pd.Series:
    x = x.str.replace("%", "")
    x = x.astype(float) / 100
    return x


def _parse_money(x: pd.Series) -> pd.Series:
    x = x.str.replace("$", "").str.replace(",", "")
    x = x.astype(float)
    return x


def preprocess_companies(companies: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the data for companies.

    Args:
        companies: Raw data.
    Returns:
        Preprocessed data, with `company_rating` converted to a float and
        `iata_approved` converted to boolean.
    """
    companies["iata_approved"] = _is_true(companies["iata_approved"])
    companies["company_rating"] = _parse_percentage(companies["company_rating"])
    return companies


def preprocess_shuttles(shuttles: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the data for shuttles.

    Args:
        shuttles: Raw data.
    Returns:
        Preprocessed data, with `price` converted to a float and `d_check_complete`,
        `moon_clearance_complete` converted to boolean.
    """
    shuttles["d_check_complete"] = _is_true(shuttles["d_check_complete"])
    shuttles["moon_clearance_complete"] = _is_true(shuttles["moon_clearance_complete"])
    shuttles["price"] = _parse_money(shuttles["price"])
    return shuttles


def create_model_input_table(
        shuttles: pd.DataFrame, companies: pd.DataFrame, reviews: pd.DataFrame
) -> pd.DataFrame:
    """Combines all data to create a model input table.

    Args:
        shuttles: Preprocessed data for shuttles.
        companies: Preprocessed data for companies.
        reviews: Raw data for reviews.
    Returns:
        Model input table.

    """
    rated_shuttles = shuttles.merge(reviews, left_on="id", right_on="shuttle_id")
    rated_shuttles = rated_shuttles.drop("id", axis=1)
    model_input_table = rated_shuttles.merge(
        companies, left_on="company_id", right_on="id"
    )
    model_input_table = model_input_table.dropna()
    return model_input_table
