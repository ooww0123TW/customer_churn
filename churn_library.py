# library doc string
"""
churn_library.py
- library of functions to find customers who are likely to churn.

Author: Wonseok
Date: April 2023
"""

# import libraries
import os

from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from pandas.plotting import table

import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


os.environ['QT_QPA_PLATFORM'] = 'offscreen'



def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            d_f: pandas dataframe
    '''
    d_f = pd.read_csv(pth)
    return d_f
    

def perform_eda(d_f):
    '''
    perform eda on d_f and save figures to images folder
    input:
            d_f: pandas dataframe

    output:
            None
    '''
    d_f.isnull().sum()
    desc = d_f.describe()

    # create a subplot without a frame
    plot = plt.subplot(111, frame_on=False)
    plot.xaxis.set_visible(False)
    plot.yaxis.set_visible(False)
    table(plot, desc, loc='upper right')
    plt.savefig('./images/eda/desc_plot.png')

    d_f['Churn'] = d_f['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    plt.figure(figsize=(20, 10))
    d_f['Churn'].hist()
    plt.savefig('./images/eda/churn_hist.png')

    plt.figure(figsize=(20, 10))
    d_f['Customer_Age'].hist()
    plt.savefig('./images/eda/customer_age_hist.png')

    plt.figure(figsize=(20, 10))
    d_f.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig('./images/eda/marital_status.png')

    plt.figure(figsize=(20, 10))
    # distplot is deprecated. Use histplot instead
    # sns.distplot(df['Total_Trans_Ct']);
    # Show distributions of 'Total_Trans_Ct' and add a smooth curve obtained
    # using a kernel density estimate
    sns.histplot(d_f['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig('./images/eda/total_trans_ct.png')

    plt.figure(figsize=(20, 10))
    sns.heatmap(d_f.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig('./images/eda/heatmap.png')


def encoder_helper(d_f, category_lst):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            d_f: pandas dataframe
            category_lst: list of columns that contain categorical features

    output:
            new_df: pandas dataframe with new columns for
    '''
    new_df = pd.DataFrame()
    for category in category_lst:
        temp_lst = []
        temp_groups = d_f.groupby(category).mean()['Churn']
        for val in d_f[category]:
            temp_lst.append(temp_groups.loc[val])

        d_f[category + '_Churn'] = temp_lst

    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    new_df[keep_cols] = d_f[keep_cols]
    return new_df


def perform_feature_engineering(d_f, response):
    '''
    input:
              d_f: pandas dataframe
              response: string of response name [optional argument
                      that could be used for naming variables or index y column]

    output:
              input_train: input training data
              input_test: input testing data
              y_train: y training data
              y_test: y testing data
    '''
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]
    d_f = encoder_helper(d_f, cat_columns)
    input_train, input_test, y_train, y_test = train_test_split(
        d_f, response, test_size=0.3, random_state=42)
    return input_train, input_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                preds):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            preds: dictionary with below keys
            - 'y_train_preds_lr': training predictions from logistic regression
            - 'y_train_preds_rf': training predictions from random forest
            - 'y_test_preds_lr': test predictions from logistic regression
            - 'y_test_preds_rf': test predictions from random forest

    output:
             None
    '''
    # scores
    print('random forest results')
    print('test results')
    print(classification_report(y_test, preds['y_test_preds_rf']))
    print('train results')
    print(classification_report(y_train, preds['y_train_preds_rf']))

    print('logistic regression results')
    print('test results')
    print(classification_report(y_test, preds['y_test_preds_lr']))
    print('train results')
    print(classification_report(y_train, preds['y_train_preds_lr']))

    plt.figure()
    plt.rc('figure', figsize=(5, 5))
    # plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, preds['y_test_preds_rf'])), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, preds['y_train_preds_rf'])), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig("./images/results/random_forest.png")

    plt.figure()
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, preds['y_train_preds_lr'])), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, preds['y_test_preds_lr'])), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig("./images/results/logistic_regression.png")


def feature_importance_plot(model, input_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            input_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''

    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [input_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(input_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(input_data.shape[1]), names, rotation=90)
    plt.savefig(output_pth)


def train_models(input_train, input_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              input_train: X training data
              input_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference:
    # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(input_train, y_train)

    lrc.fit(input_train, y_train)
    preds = dict.fromkeys(['y_train_preds_lr', 'y_train_preds_rf',
                           'y_test_preds_lr', 'y_test_preds_rf'])
    preds['y_train_preds_rf'] = cv_rfc.best_estimator_.predict(input_train)
    preds['y_test_preds_rf'] = cv_rfc.best_estimator_.predict(input_test)

    preds['y_train_preds_lr'] = lrc.predict(input_train)
    preds['y_test_preds_lr'] = lrc.predict(input_test)

    lrc_plot = plot_roc_curve(lrc, input_test, y_test)

    # plots
    plt.figure(figsize=(15, 8))
    axs = plt.gca()
    plot_roc_curve(
        cv_rfc.best_estimator_,
        input_test,
        y_test,
        ax=axs,
        alpha=0.8)
    lrc_plot.plot(ax=axs, alpha=0.8)
    plt.savefig("./images/results/roc_curve.png")

    # save best model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    plt.figure()
    explainer = shap.TreeExplainer(cv_rfc.best_estimator_)
    shap_values = explainer.shap_values(input_test)
    shap.summary_plot(shap_values, input_test, plot_type="bar", show=False)
    plt.savefig("./images/results/shap.png")

    classification_report_image(
        y_train,
        y_test,
        preds)

    feature_importance_plot(
        cv_rfc.best_estimator_,
        input_train,
        "./images/results/feature_importance.png")


if __name__ == "__main__":
    data_frame = import_data(r"./data/bank_data.csv")
    perform_eda(data_frame)
    x_train, x_test, churn_train, churn_test = perform_feature_engineering(
        data_frame, data_frame['Churn'])
    train_models(x_train, x_test, churn_train, churn_test)
