"""
Carlos Yu
20SU CSE 163 AC

This program predicts the us covid-19 cases growth in the early August
and optimize the model through controlling the max depth of the decision
tree regressor and the number of features to be included
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error


DATA_US = "/home/datasets/us_covid19_daily.csv"
DATA_WORLD = "/home/datasets/COVID-19.csv"
# This list comes from the result of RQ1
FILTER_LIST = ['date',
               'parks_percent_change_from_baseline',
               'residential_percent_change_from_baseline',
               'grocery_and_pharmacy_percent_change_from_baseline',
               'retail_and_recreation_percent_change_from_baseline',
               'transit_stations_percent_change_from_baseline',
               'workplaces_percent_change_from_baseline']
MIN_DEPTH = 4
MAX_DEPTH = 15


def remove_dashes(dashed_date):
    """
    remove the dashes from a 'yyyy-mm-dd' formatted date
    write as a separate function because the apply(lambda x) one doesn't
    work as expected
    """
    return str(dashed_date).replace('-', '')


def cut_year(plain_date):
    """
    cut out the first four year number from a 'yyyymmdd' formatted date
    write as a separate function because the apply(lambda x) one doesn't
    work as expected
    """
    return plain_date[4:]


def load_data():
    """
    Parses the two datasets, filter them to leave only the columns needed
    return the two filtered dataframes
    """
    us_data = pd.read_csv(DATA_US)
    us_data = us_data[['date', 'positiveIncrease']].fillna(0)
    world_data = pd.read_csv(DATA_WORLD)
    world_data = world_data[world_data['country'] == 'United States']
    world_data = world_data[FILTER_LIST].fillna(0)
    return us_data, world_data


def optimize_model(us, world):
    """
    Takes the two cleaned dataframes
    feed the model every possible max depth of the decision tree regressor
    within the given range and every possible number of features to be
    included, record the combination that yields the least mean squared error
    return the best tree depth, best number of features, and the error
    """
    min_error = float('inf')
    best_depth = MIN_DEPTH
    best_features = len(FILTER_LIST)
    for i in range(MIN_DEPTH, MAX_DEPTH):
        cur = predict(us, world, i, len(FILTER_LIST), False)
        if cur < min_error:
            best_depth = i
            min_error = cur
    # 2 for the lower bound since the first one is the must-include 'date'
    # column, and there needs to be at least one feature
    # +1 for the upper bound to be able to include everything
    for j in range(2, len(FILTER_LIST)+1):
        cur = predict(us, world, best_depth, j, False)
        if cur < min_error:
            best_features = j
            min_error = cur
    return best_depth, best_features, min_error


def predict(us, world, tree_depth, feature_num, if_graph):
    """
    Takes the filtered two datasets, an integer indicating the maximum depth
    of the decision tree model, an integer indicating the number of features
    to be included, and a boolean indicating whether to plot the result

    Merge the two datasets by the date, and train a decision tree regressor
    to predict the later 15% of the dates (~Jul 10 to Aug 12) based on the
    results, calculate the mean squared error between the real data and the
    prediction, and return this error

    if the boolean parameter is true, plot a double-line graph comparing the
    real data and the prediction and save it as "us_new_cases.png"
    """
    if feature_num < len(FILTER_LIST):
        world = world[FILTER_LIST[0:feature_num]]
    world['date'] = world['date'].apply(remove_dashes)
    us['date'] = us['date'].apply(remove_dashes)
    comb = us.merge(world, how='left', left_on='date', right_on='date')
    comb = comb.fillna(0)
    comb['date'] = comb['date'].apply(cut_year)

    features = comb.loc[:, comb.columns != 'positiveIncrease']
    labels = comb['positiveIncrease']
    features_test, features_train = np.split(features,
                                             [int(.225 * len(features))])
    labels_test, labels_train = np.split(labels, [int(.225 * len(labels))])

    model = DecisionTreeRegressor(max_depth=tree_depth)
    model.fit(features_train, labels_train)
    test_predictions = model.predict(features_test)
    test_err = mean_squared_error(labels_test, test_predictions)

    if if_graph:
        fig, ax = plt.subplots(1)
        df1 = comb[['date', 'positiveIncrease']]
        df1 = df1[::-1]
        lis = list(test_predictions) + list(labels_train)
        df2 = pd.concat([comb['date'], pd.Series(lis, name='prediction')],
                        axis=1)
        df2 = df2[::-1]
        fig, ax = plt.subplots(1)
        df1.plot(ax=ax, x='date', y='positiveIncrease', kind='line')
        df2.plot(ax=ax, x='date', y='prediction', kind='line')
        plt.xlabel('Date')
        plt.ylabel('New Cases')
        plt.title("Covid-19 New Cases in the U.S")
        plt.savefig('/home/plots/us_new_cases.png', bbox_inches='tight')

    return test_err


def main():
    us_data, world_data = load_data()
    depth, features, error = optimize_model(us_data, world_data)
    predict(us_data, world_data, depth, features, True)


if __name__ == '__main__':
    main()
