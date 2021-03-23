"""
Joshua Zhang
CSE 163 AC

This file includes 9 functions for helping answer question2
of the final project in cse 163 class. This part of question
includes the relationship between several factors such as
population densities or mobility change's influence on the
spread of Covid-19. 4 of the functions intend to filter and
clean the data before use to avoid similar steps of work.
5 of the functions output and save graphs in different forms.
"""

# Import packages
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns

# eliminate package warnings and set environment for graphs
pd.options.mode.chained_assignment = None
sns.set()


def covid_data_clean(data):
    '''
    This functions takes in covid-19 data and filters it until
    only date, state, and positive cases information is included.
    Also, it transform date data such as '20200305' into '3.05'
    for easier visualization.
    It returns a new dataframe.
    '''
    cleaned = data[['date', 'state', 'positive']]
    # transform date data for better visualization
    cleaned.loc[:, 'date'] = (cleaned['date'] - 20200000) / 100
    cleaned.dropna()
    return cleaned


def population_density_clean(population_data, shapes):
    '''
    This functions takes in USA population data and USA shapes
    in order to give the orginal population data state abbrevations.
    Then it sorts the data by population density.
    It returns a new population density dataframe that has state
    abbrevations which allow further comparasion between population
    data and other data.
    '''
    filtered_state = shapes[['STATE_NAME', 'STATE_ABBR']]
    population_density = filtered_state.merge(
                        population_data,
                        left_on='STATE_NAME',
                        right_on='State', how='right')
    population_density = population_density.sort_values(by='Density')
    population_density.dropna()
    return population_density


def total_cases_each_state(data):
    '''
    This functions takes in Covid-19 cases data and returns a new
    dataframe with each state with their total cases.
    '''
    state = data.groupby(['state'])['positive'].max()
    state = state.to_frame()
    state.reset_index(inplace=True)
    state.dropna()
    return state


def total_cases_each_state_plot(shapes, state):
    '''
    This function takes in usa shapes data and total
    cases each state in order to outputs a map of
    Covid-19 cases in USA. The cases are highlighted
    by red color to show high level of seriousness.
    '''
    merged = shapes.merge(state, left_on='STATE_ABBR',
                          right_on='state', how='left')
    merged.plot(column='positive', cmap='coolwarm', legend=True)
    plt.title('Map of Covid-19 Cases per State in USA')
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    plt.savefig('plots/total_case_each_state.png')


def largest_lowest(state):
    '''
    This function takes in total cases each state and
    returns a list of two lists.The first list includes
    10 states with the least Covid-19 cases when the
    second one has ten states with the most Covid-19
    cases. The states are described by their abbrevations.
    Notification: 4 states with the least cases are dropped
    snice they are not states when they are territory of
    USA. For example, Guam.
    '''
    state = state.sort_values(by='positive')
    state.reset_index(inplace=True)
    state.drop(state.head(4).index, inplace=True)
    lowest = state.head(10)
    state_lowest = list(lowest['state'])
    largest = state.tail(10)
    state_largest = list(largest['state'])
    result = [state_lowest, state_largest]
    return result


def largest_lowest_plot(data, large_low):
    '''
    This function takes in USA Covid-19 data and abbrevations
    of states with the least and most cases. It outputs two line
    graphs with the daily growth of these most and least cases
    state. The final picture is saved in
    'plots/largest_lowest_states_daily_growth.png'
    '''
    fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 10))
    filtered_low = pd.DataFrame()
    filtered_high = pd.DataFrame()
    for state in large_low[0]:
        filtered_low = filtered_low.append(data[data['state'] == state])
    for state in large_low[1]:
        filtered_high = filtered_high.append(data[data['state'] == state])
    sns.lineplot(x="date", y="positive", data=filtered_low,
                 hue='state', ax=ax1)
    ax1.set_title('Daily Growth of 10 States with the Least Cases')
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Cases')
    sns.lineplot(x="date", y="positive", data=filtered_high,
                 hue='state', ax=ax2)
    ax2.set_title('Daily Growth of 10 States with the Most Cases')
    ax2.set_xlabel('Month')
    ax2.set_ylabel('Cases')
    plt.subplots_adjust(hspace=0.2)
    plt.savefig('plots/largest_lowest_states_daily_growth.png',
                bbox_inches="tight")


def population_density_factor_plot(population_data, large_low):
    '''
    This function takes in population density data and abbrevations
    of states with the least and most cases. It outputs two bar
    graphs of population densities of each of these states and
    their averages.The final picture is saved in
    'plots/largest_lowest_states_population_density.png'
    '''
    fig, (ax1, ax2) = plt.subplots(2)
    filtered_low = pd.DataFrame()
    filtered_high = pd.DataFrame()
    for state in large_low[0]:
        filtered_low = filtered_low.append(
            population_data[population_data['STATE_ABBR'] == state])
    filtered_low = filtered_low.append(
        {'STATE_ABBR': 'Average', 'Density': filtered_low['Density'].mean()},
        ignore_index=True)
    for state in large_low[1]:
        filtered_high = filtered_high.append(
            population_data[population_data['STATE_ABBR'] == state])
    filtered_high = filtered_high.append(
        {'STATE_ABBR': 'Average',
         'Density': filtered_high['Density'].mean()},
        ignore_index=True)
    sns.barplot(x='STATE_ABBR', y='Density', data=filtered_low, ax=ax1)
    ax1.set_title('Population Density of 10 States with the Least Cases')
    ax1.set_ylabel('People per Sqd miles')
    ax1.set_xlabel('States')
    ax1.text(9.5, 50, '59.64')
    sns.barplot(x='STATE_ABBR', y='Density', data=filtered_high, ax=ax2)
    ax2.set_title('Population Density of 10 States with the Most Cases')
    ax2.set_ylabel('People per Sqd miles')
    ax2.set_xlabel('States')
    ax2.text(9.5, 350, '321.25')
    plt.subplots_adjust(hspace=0.5)
    plt.savefig('plots/largest_lowest_states_population_density.png',
                bbox_inches="tight")


def hospital_factor_plot(hospital_data, large_low, population_data):
    '''
    This function takes in hospital data, population data and
    abbrevations of states with the least and most cases.
    It outputs two bar graphs of population divided by number
    of hospitals of each of these states and
    their averages.The final picture is saved in
    'plots/largest_lowest_states_hospital_proportion.png'
    '''
    fig, (ax1, ax2) = plt.subplots(2)
    filtered_hospital = hospital_data.groupby(['STATE']).size()
    filtered_hospital = filtered_hospital.to_frame()
    filtered_hospital = filtered_hospital.rename(columns={0: 'Hospital'})
    merged = filtered_hospital.merge(population_data, left_on='STATE',
                                     right_on='STATE_ABBR')
    merged['Proportion'] = merged['Pop'] / merged['Hospital']
    filtered_low = pd.DataFrame()
    filtered_high = pd.DataFrame()
    for state in large_low[0]:
        filtered_low = filtered_low.append(
            merged[merged['STATE_ABBR'] == state])
    filtered_low = filtered_low.append(
        {'STATE_ABBR': 'Average',
         'Proportion': filtered_low['Proportion'].mean()},
        ignore_index=True)
    for state in large_low[1]:
        filtered_high = filtered_high.append(
            merged[merged['STATE_ABBR'] == state])
    filtered_high = filtered_high.append(
        {'STATE_ABBR': 'Average',
         'Proportion': filtered_high['Proportion'].mean()},
        ignore_index=True)
    sns.barplot(x='STATE_ABBR', y='Proportion', data=filtered_low, ax=ax1)
    ax1.set_title(
        'People to Hospital Proportion of 10 States with the Least Cases')
    ax1.set_ylabel('People per Hospital')
    ax1.set_xlabel('States')
    ax1.text(9.3, 23000, '25172.37')
    sns.barplot(x='STATE_ABBR', y='Proportion', data=filtered_high, ax=ax2)
    ax2.set_title(
        'People to Hospital Proportion of 10 States with the Most Cases')
    ax2.set_ylabel('People per Hospital')
    ax2.set_xlabel('States')
    ax2.text(9.3, 52000, '54001.81')
    plt.subplots_adjust(hspace=0.5)
    plt.savefig('plots/largest_lowest_states_hospital_proportion.png',
                bbox_inches="tight")


def mobility_factor_plot(mobility_data, large_low):
    '''
    This function takes in USA mobility change data(Feb to Apr)
    and abbrevations of states with the least and most cases.
    It outputs two bar graphs of average mobility change of
    six specific types of working areas of each of these states
    and their averages.The final picture is saved in
    'plots/largest_lowest_states_average_mobility_change.png'
    '''
    fig, (ax1, ax2) = plt.subplots(2)
    mobility_data['Average'] = mobility_data.mean(axis=1, numeric_only=True)
    filtered_low = pd.DataFrame()
    filtered_high = pd.DataFrame()
    for state in large_low[0]:
        filtered_low = filtered_low.append(
            mobility_data[mobility_data['Abbrev'] == state])
    filtered_low = filtered_low.append(
        {'Abbrev': 'Average', 'Average': filtered_low['Average'].mean()},
        ignore_index=True)
    for state in large_low[1]:
        filtered_high = filtered_high.append(
            mobility_data[mobility_data['Abbrev'] == state])
    filtered_high = filtered_high.append(
        {'Abbrev': 'Average', 'Average': filtered_high['Average'].mean()},
        ignore_index=True)
    sns.barplot(x='Abbrev', y='Average', data=filtered_low, ax=ax1)
    ax1.set_title(
        'Average Mobility Change of States with the Least Cases(Feb to Apr)')
    ax1.set_ylabel('Mobility Change')
    ax1.set_xlabel('States')
    ax1.text(9.6, -22, '-20.5')
    sns.barplot(x='Abbrev', y='Average', data=filtered_high, ax=ax2)
    ax2.set_title(
        'Average Mobility Change of States with the Most Cases(Feb to Apr)')
    ax2.set_ylabel('Mobility Change')
    ax2.set_xlabel('States')
    ax2.text(9.6, -30.3, '-29.45')
    plt.subplots_adjust(hspace=0.5)
    plt.savefig('plots/largest_lowest_states_average_mobility_change.png',
                bbox_inches="tight")


def main():
    # load in data
    shapes = gpd.read_file('/home/datasets/states_21basic/states.shp')
    data = pd.read_csv('/home/datasets/us_states_covid19_daily.csv')
    population_data = pd.read_csv('/home/datasets/USA_population.csv')
    hospital_data = pd.read_csv('/home/datasets/Hospitals.csv')
    mobility_data = pd.read_csv(
        '/home/datasets/datasets_COVID19_Google_Mobility_Report_US_State.csv')
    # clean data
    data = covid_data_clean(data)
    population_data = population_density_clean(population_data, shapes)
    state = total_cases_each_state(data)
    large_low = largest_lowest(state)
    # ploting
    total_cases_each_state_plot(shapes, state)
    largest_lowest_plot(data, large_low)
    population_density_factor_plot(population_data, large_low)
    hospital_factor_plot(hospital_data, large_low, population_data)
    mobility_factor_plot(mobility_data, large_low)


if __name__ == '__main__':
    main()
