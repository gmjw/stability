import os

import pandas as pd
from pytest import fixture, mark


from src.stability.decorators import stability_test


PARAMS_DFS = [
    pd.DataFrame({
        'str_col': ['a', 'b'],
        'int_col': [1, 2],
    }),
    pd.DataFrame({
        'float_col': [3.14159, -1],
    }),
]


@fixture
def my_fixture():
    return pd.DataFrame({'col': [10, 20, 30]})


def get_all_dtypes_df():
    df = pd.DataFrame({
        'str_col': ['a', 'b'],
        'int_col': [1, 2],
        'float_col': [3.14159, -1],
        'datetime_col': ['2000-01-01 00:00:00', '2001-01-01 12:34:56'],
    })
    df['datetime_col'] = pd.to_datetime(df['datetime_col'])  # [ns]
    df['datetime_col2'] = df['datetime_col'].dt.tz_localize('UTC')  # [ns, UTC]
    return df


@stability_test
def test_decorator_basics():
    df = get_all_dtypes_df()
    return df


@stability_test
def test_decorator_with_fixture(my_fixture):
    out = 2 * my_fixture
    return out


@stability_test(filetype='pq')
def test_decorator_parquet_file(my_fixture):
    df = get_all_dtypes_df()
    return df


@stability_test(filetype='json')
def test_decorator_json_file(my_fixture):
    df = get_all_dtypes_df()
    return df


@stability_test
def test_decorator_yielding():
    for df in PARAMS_DFS:
        yield df


@mark.parametrize(
    ['test_case', 'df'],
    enumerate(PARAMS_DFS),
)
def test_decorator_parametrized(test_case, df):

    @stability_test(test_case=test_case)
    def test_decorator_parametrized_inner():
        return df

    test_decorator_parametrized_inner()


@mark.parametrize(
    'df',
    PARAMS_DFS,
    ids=['case0', 'case1']
)
def test_decorator_parametrized_with_ids(df, request):

    @stability_test(test_case=request.node.callspec.id)
    def test_decorator_parametrized_with_ids_inner():
        return df

    test_decorator_parametrized_with_ids_inner()



@mark.xfail  # output is not a dataframe
@stability_test
def test_decorator_output_not_a_dataframe():
    return 42


@mark.xfail  # output index is not 0...n-1
@stability_test
def test_decorator_output_index_invalid():
    return pd.DataFrame({
        'float_col': [3.14159, -1],
    }, index=['a', 'b'])


@stability_test(filetype='pq')
def test_decorator_non_range_index_fine_when_using_parquet():
    return pd.DataFrame({
        'float_col': [3.14159, -1],
    }, index=['a', 'b'])
