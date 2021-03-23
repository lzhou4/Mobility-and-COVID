"""
Angel Zhou
CSE163 AC

This file contains functions used to produce graphs
and calculations to help answer the research question 1
"""


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()


def load_data():
    df = pd.read_csv('/home/datasets/COVID-19.csv')
    df = df[df['iso'] == 'US']
    df = df.loc[:, 'date':'confirmed_cases']
    s = pd.Series([i for i in range(len(df))])
    df = df.set_index(s)
    df = df.loc[0:163]
    return df


def plot_daily_new(df):
    """
    Takes a dataframe as a parameter and plots the daily new confirmed
    cases over time in the US and saves the plot as
    '/home/plots/daily_new.png'
    """
    fig, ax = plt.subplots(1)
    df.loc[:, 'daily_new_confirmed'] = df['confirmed_cases'].diff()
    sns.lineplot(x=df.index, y='daily_new_confirmed', data=df, ax=ax)
    plt.xlabel('Days since 02/15/2020')
    plt.ylabel('New Cases')
    plt.title('Daily New Confirmed Cases', fontweight="bold")
    fig.savefig('/home/plots/daily_new.png', bbox_inches='tight')


def plot_locations(df):
    """
    Takes a dataframe as a parameter and plots the mobility percent change
    from baseline over time for various locations and saves the plot as
    '/home/plots/locations_vs_day.png'
    """
    fig, [[ax1, ax2], [ax3, ax4], [ax5, ax6]] = plt.subplots(3, ncols=2,
                                                             figsize=(20, 15))
    sns.lineplot(y='grocery_and_pharmacy_percent_change_from_baseline',
                 x=df.index, data=df, ax=ax1)
    sns.lineplot(y='parks_percent_change_from_baseline',
                 x=df.index, data=df, ax=ax2)
    sns.lineplot(y='residential_percent_change_from_baseline',
                 x=df.index, data=df, ax=ax3)
    sns.lineplot(y='retail_and_recreation_percent_change_from_baseline',
                 x=df.index, data=df, ax=ax4)
    sns.lineplot(y='transit_stations_percent_change_from_baseline',
                 x=df.index, data=df, ax=ax5)
    sns.lineplot(y='workplaces_percent_change_from_baseline',
                 x=df.index, data=df, ax=ax6)
    ax1.set_title('Grocery and Pharmacy', fontweight="bold", size=20)
    ax2.set_title('Parks', fontweight="bold", size=20)
    ax3.set_title('Residential', fontweight="bold", size=20)
    ax4.set_title('Retail and Recreation', fontweight="bold", size=20)
    ax5.set_title('Transit Stations', fontweight="bold", size=20)
    ax6.set_title('Workplace', fontweight="bold", size=20)
    ax5.set_xlabel('Days since 02/15/2020', fontsize=18)
    ax6.set_xlabel('Days since 02/15/2020', fontsize=18)
    axes = [ax1, ax2, ax3, ax4, ax5, ax6]
    for i in axes:
        i.set_ylabel('Percentage change from baseline', fontsize=18)
    fig.savefig('/home/plots/locations_vs_day.png', bbox_inches='tight')


def correlation(df):
    """
    Takes a dataframe as a parameter, calculates the correlation between
    daily new confirmed cases and mobility change, then returns a list of
    dictionary sorted by the correlation in descending order
    """
    locations = ['grocery_and_pharmacy_percent_change_from_baseline',
                 'parks_percent_change_from_baseline',
                 'residential_percent_change_from_baseline',
                 'retail_and_recreation_percent_change_from_baseline',
                 'transit_stations_percent_change_from_baseline',
                 'workplaces_percent_change_from_baseline']
    result = {}
    for location in locations:
        result[location.split('_')[0]] = round(df[location].corr(
                                         df['daily_new_confirmed']), 4)
    result = sorted(result.items(), key=lambda x: x[1], reverse=True)
    for pair in result:
        print(pair[0] + ': ' + str(pair[1]))


def main():
    df = load_data()
    plot_daily_new(df)
    plot_locations(df)
    print('Correlation between mobility and new cases:')
    correlation(df)


if __name__ == '__main__':
    main()
