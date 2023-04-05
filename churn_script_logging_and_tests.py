"""
Author: Wonseok Oh
Date: April 2023

churn_script_logging_and_tests.py
"""

import os
import logging
import churn_library as cls

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test data import - this example is completed for you
        to assist with the other test functions
    '''
    try:
        d_f = import_data("./data/bank_data.csv")
    except FileNotFoundError as err:
        logging.error("ERROR: Testing import_data: File not found: %s", "./data/bank_data.csv")
        raise err

    try:
        assert d_f.shape[0] > 0
        assert d_f.shape[1] > 0
    except AssertionError as err:
        logging.error("ERROR: Testing import_data: The file doesn't appear\
                      to have rows and columns")
        raise err

    logging.info("Testing import_data: SUCCESS")
    return d_f


def test_eda(perform_eda, d_f):
    '''
    test perform eda function
    '''
    perform_eda(d_f)
    try:
        assert os.path.isfile("./images/eda/desc_plot.png")
        assert os.path.isfile("./images/eda/churn_hist.png")
        assert os.path.isfile("./images/eda/customer_age_hist.png")
        assert os.path.isfile("./images/eda/marital_status.png")
        assert os.path.isfile("./images/eda/total_trans_ct.png")
        assert os.path.isfile("./images/eda/heatmap.png")
        logging.info("Testing perform_eda: SUCCESS")
    except AssertionError as err:
        logging.error("ERROR: Testing perform_eda: Output files don't exist")
        raise err


def test_encoder_helper(encoder_helper, d_f):
    '''
    test encoder helper
    '''
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]

    new_df = encoder_helper(d_f, cat_columns)
    try:
        assert new_df.shape[0] > 0
        assert new_df.shape[1] > 0
        logging.info("Testing encoder_helper: SUCCESS")
    except AssertionError as err:
        logging.error("ERROR: Testing encoder_helper: The file doesn't appear\
                      to have rows and columns")
        raise err
    return new_df


def test_perform_feature_engineering(perform_feature_engineering,
                                     d_f):
    '''
    test perform_feature_engineering
    '''
    x_train, x_test, churn_train, churn_test = perform_feature_engineering(
        d_f, d_f['Churn'])

    try:
        assert x_train.shape[0] > 0
        assert x_train.shape[1] > 0
        assert x_test.shape[0] > 0
        assert x_test.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "ERROR: Testing perform_feature_engineering: The file doesn't appear\
                      to have rows and columns")
        raise err

    try:
        assert x_train.shape[0] == churn_train.shape[0]
        assert x_test.shape[0] == churn_test.shape[0]
    except AssertionError as err:
        logging.error(
            "ERROR: Testing perform_feature_engineering: The dimensions don't match")
        raise err

    logging.info("Testing perform_feature_engineering: SUCCESS")
    return x_train, x_test, churn_train, churn_test


def test_train_models(train_models, x_train, x_test, churn_train, churn_test):
    '''
    test train_models
    '''
    train_models(x_train, x_test, churn_train, churn_test)

    try:
        assert os.path.isfile("./models/logistic_model.pkl")
        assert os.path.isfile("./models/rfc_model.pkl")
    except AssertionError as err:
        logging.error("ERROR: Testing test_train_models: The models \
                      do not exist")
        raise err

    try:
        assert os.path.isfile("./images/results/shap.png")
        assert os.path.isfile("./images/results/feature_importance.png")
        assert os.path.isfile("./images/results/random_forest.png")
        assert os.path.isfile("./images/results/logistic_regression.png")
    except AssertionError as err:
        logging.error("ERROR: Testing test_train_models: The result images \
                      do not exist")
        raise err

    logging.info("Testing train_models: SUCCESS")


if __name__ == "__main__":
    data_frame = test_import(cls.import_data)
    test_eda(cls.perform_eda, data_frame)
    data_frame = test_encoder_helper(cls.encoder_helper, data_frame)
    input_train, input_test, y_train, y_test = test_perform_feature_engineering(
        cls.perform_feature_engineering, data_frame)
    test_train_models(
        cls.train_models,
        input_train,
        input_test,
        y_train,
        y_test)
