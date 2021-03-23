"""
Joshua Zhang, Angel Zhou, Carlos Yu
CSE 163 AC

Tests the code on the three research questions
"""

import RQ1
import RQ2
import RQ3
import pandas as pd
from cse163_utils import assert_equals


def test_RQ1():
    """
    Tests RQ1 on its DataFrame size
    """
    df = RQ1.load_data()
    assert_equals(8, len(df.columns))
    assert_equals(164, len(df))
    print('RQ1 tests all clear')


def test_RQ2():
    """
    Tests RQ2 on its data filtering
    """
    data = pd.read_csv('datasets/us_states_covid19_daily.csv')
    state = RQ2.total_cases_each_state(data)
    states = [['VT', 'WY', 'HI', 'ME', 'AK', 'MT', 'NH', 'ND', 'WV', 'SD'],
              ['LA', 'NC', 'NJ', 'AZ', 'IL', 'GA', 'NY', 'TX', 'FL', 'CA']]
    assert_equals(states, RQ2.largest_lowest(state))
    df = {'state': [
              'AK', 'AL', 'AR', 'AS', 'AZ', 'CA', 'CO', 'CT', 'DC',
              'DE', 'FL', 'GA', 'GU', 'HI', 'IA', 'ID', 'IL', 'IN', 'KS',
              'KY', 'LA', 'MA', 'MD', 'ME', 'MI', 'MN', 'MO', 'MP', 'MS',
              'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM', 'NV', 'NY', 'OH',
              'OK', 'OR', 'PA', 'PR', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT',
              'VA', 'VI', 'VT', 'WA', 'WI', 'WV', 'WY'],
          'positive': [
              4655.0, 104786.0, 51114.0, 0.0, 189443.0, 586056.0,
              51441.0, 50706.0, 12959.0, 15699.0, 550901.0, 226153.0, 449.0,
              3958.0, 49806.0, 25595.0, 199893.0, 76522.0, 32547.0, 36945.0,
              134304.0, 122000.0, 97384.0, 4070.0, 98689.0, 62303.0, 62530.0,
              49.0, 69374.0, 5268.0, 139061.0, 7970.0, 29030.0, 6887.0,
              185938.0, 22643.0, 58048.0, 422703.0, 104248.0, 45398.0,
              22022.0, 121130.0, 24074.0, 20129.0, 102974.0, 9815.0, 126393.0,
              506820.0, 45090.0, 102521.0, 639.0, 1478.0, 64151.0, 66654.0,
              8008.0, 3086.0]}
    df = pd.DataFrame(df, columns=['state', 'positive'])
    assert_equals(True, df.equals(RQ2.total_cases_each_state(data)))
    print('RQ2 tests all clear')


def test_RQ3():
    """
    Tests RQ3 on its data filtering and model optimization
    """
    us_data, world_data = RQ3.load_data()
    # check if the data is filtered correctly
    assert_equals(2, len(us_data.columns))
    assert_equals(7, len(world_data.columns))
    assert_equals(RQ3.FILTER_LIST, world_data.columns)
    # best combination: depth = 13, features = 7, anything else will produce
    # larger error value
    assert_equals(True, RQ3.predict(us_data, world_data, 15, 6, False)
                  > 64597842.9)
    assert_equals(True, RQ3.predict(us_data, world_data, 7, 5, False)
                  > 64597842.9)
    print('RQ3 tests all clear')


def main():
    test_RQ1()
    test_RQ2()
    test_RQ3()


if __name__ == '__main__':
    main()
