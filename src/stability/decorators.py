import os
from enum import Enum
from functools import wraps
from pathlib import Path

from os.path import dirname, realpath, exists
from os import mkdir, environ
from typing import Callable, Any

import pandas as pd

from inspect import isgeneratorfunction, getfile


WRITE_DATA = bool(environ.get('STABILITY_WRITE_DATA'))


def convert_datetime_columns_to_strings(df):
    """
    When using CSV or JSON filetypes, we must convert
    datetime columns to strings.
    This is because:
    - for CSVs, reading from disk and knowing which columns
      are datetime columns is awkward.
    - timestamps are not JSON-serializable.

    This means that the decorator cannot pick up on cases
    where the dtype of one of the columns being checked has
    changed from datetime64 to str, but this is likely to be
    very rare, and the user has been warned about this in the
    README.  # TODO: Mention this!
    """
    dtypes = ["datetime64", "datetimetz"]
    datetime_df = df.select_dtypes(dtypes)
    datetime_cols = list(datetime_df.columns)

    if datetime_cols:
        # TODO: Log what's happening here so users have some visibility
        df = df.astype({c: str for c in datetime_cols})

    return df


class CsvFileHandler:
    @staticmethod
    def write_to_disk(df, filepath):
        df.to_csv(filepath, index=False)

    @staticmethod
    def load_from_disk(filepath):
        return pd.read_csv(filepath)

    @staticmethod
    def convert(df):
        return convert_datetime_columns_to_strings(df)


class ParquetFileHandler:
    @staticmethod
    def write_to_disk(df, filepath):
        df.to_parquet(filepath)

    @staticmethod
    def load_from_disk(filepath):
        return pd.read_parquet(filepath)

    @staticmethod
    def convert(df):
        return df  # No conversions necessary



class JsonFileHandler:
    @staticmethod
    def write_to_disk(df, filepath):
        df.to_json(filepath, orient='records', indent=2)

    @staticmethod
    def load_from_disk(filepath):
        return pd.read_json(filepath, orient='records')

    @staticmethod
    def convert(df):
        return convert_datetime_columns_to_strings(df)



class FileHandlers(Enum):
    csv = CsvFileHandler()
    parquet = ParquetFileHandler()
    pq = ParquetFileHandler()
    json = JsonFileHandler()
    js = JsonFileHandler()


class RememberError(Exception):
    pass


def stability_test(
    func=None,
    write: bool = False,
    filetype: str = 'csv',
    test_case: str = None,
    **kwargs,
):
    if isinstance(func, Callable):
        # Has been called "directly":
        #   @stability_test
        #   def test_...():
        return get_wrapped_func(func, write, filetype, test_case, **kwargs)
    else:
        # Has been called with arguments:
        #   @stability_test(test_case=...)
        #   def test_...():
        def dec(f):
            return get_wrapped_func(f, write, filetype, test_case, **kwargs)
        return dec


def get_wrapped_func(
    func: Callable,
    write: bool = False,
    filetype: str = 'csv',
    test_case=None,
    **dec_kwargs,
):
    write = write or WRITE_DATA

    @wraps(func)
    def wrapped(*func_args, **func_kwargs):
        out = func(*func_args, **func_kwargs)

        if isgeneratorfunction(func):
            for i, yielded in enumerate(out):
                assert_output_equals_expected(
                    yielded,
                    func,
                    write,
                    filetype,
                    test_case=i,
                    **dec_kwargs,
                )
        else:
            assert_output_equals_expected(
                out,
                func,
                write,
                filetype,
                test_case=test_case,
                **dec_kwargs,
            )

        if write:
            reminder = (
                'Everything is fine, this error is just here to remind you to '
                'switch off `write=True` before committing your changes.'
            )
            raise RememberError(reminder)

    return wrapped


def assert_output_equals_expected(
    out: pd.DataFrame,
    func: Callable,
    write: bool,
    filetype: str,
    test_case: int | str,
    **dec_kwargs,
):
    run_output_checks(out)

    assert filetype in FileHandlers.__members__, f"{filetype=} is not a valid filetype"
    handler = FileHandlers[filetype].value
    out = handler.convert(out)

    filepath = get_expected_csv_filepath(func, filetype, test_case)

    if write:
        handler.write_to_disk(out, filepath)

    expected = handler.load_from_disk(filepath)
    pd.testing.assert_frame_equal(out, expected, **dec_kwargs)


def run_output_checks(out):
    error = (
        "Only dataframes can be used with the `stability_test` decorator, "
        f"found output of type {type(out)}."
    )
    assert isinstance(out, pd.DataFrame), error

    error = (
        "Dataframes for use with the `stability_test` decorator should "
        f"always have a 0...n-1 range index, found {out.index}."
    )
    valid_index = pd.RangeIndex(len(out))
    assert out.index.equals(valid_index), error


def convert_dtypes(df):
    """
    Because we will compare to the output of a CSV file,
    we convert timestamps to strings. This means that the
    decorator cannot pick up on cases where the dtype of
    one of the columns being checked has changed from
    datetime64 to str, but the user has been warned
    about this in the README.
    """
    dtypes = ["datetime64", "datetimetz"]
    datetime_df = df.select_dtypes(dtypes)
    date_columns = datetime_df.columns

    out = df.astype({c: str for c in date_columns})
    return out


def get_expected_csv_filepath(func, filetype: str, test_case=None):
    test_filepath = getfile(func)
    test_filename = test_filepath.split(os.sep)[-1].split('.')[0]

    folder = full_path(test_filepath) / 'resources'
    if not exists(folder):
        mkdir(folder)

    test_name = func.__name__

    suffix = "" if test_case is None else f"_{test_case}"
    filename = f"{test_filename}_{test_name}{suffix}.{filetype}"

    filepath = folder / filename
    return filepath


def full_path(file: str) -> Path:
    return Path(dirname(realpath(file)))
