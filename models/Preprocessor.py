from typing import Tuple, Any
from multipledispatch import dispatch
from pandas import DataFrame, Series
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


class Preprocessor:
    @staticmethod
    def features_target(df: DataFrame) -> tuple[DataFrame, Series]:
        y = df['future_price_1']
        x = df.drop(['future_price_1'], axis=True)
        return x, y

    @staticmethod
    def train_test_split(df: DataFrame, test_size: float = 0.33) -> tuple[DataFrame, Series, DataFrame, Series]:
        if test_size > 1 or test_size < 0:
            raise ValueError("test_size should be between 0 and 1")

        train_end = int(len(df) * (1 - test_size))

        x, y = Preprocessor.features_target(df)
        x_train = x.iloc[:train_end]
        y_train = y.iloc[:train_end]
        x_test = x.iloc[train_end:]
        y_test = y.iloc[train_end:]

        return x_train, y_train, x_test, y_test

    @staticmethod
    def init_data(data_list: list) -> DataFrame:
        df = DataFrame(
            data=data_list,
            dtype=float,
            columns=[
                'open_time', 'open_price', 'high_price', 'low_price', 'close_price', 'volume',
                'close_time', 'quote', 'trades_num', 'taker_buy_base', 'taker_buy_quote', 'ignore'
            ]
        )
        return df

    @staticmethod
    def cleared_data(df: DataFrame) -> DataFrame:
        return df[
            ['close_price', 'open_price', 'high_price', 'low_price', 'volume', 'trades_num', 'quote', 'taker_buy_base',
             'taker_buy_quote']]

    @staticmethod
    def min_max_scale(df: DataFrame) -> MinMaxScaler:
        scaler = MinMaxScaler()
        scaler.fit(df)
        return scaler

    @staticmethod
    def prev_close(df: DataFrame, shift=1) -> DataFrame:
        if shift <= 0:
            raise ValueError("Shift can not be negative or equals zero")

        df = df.copy()
        df[f'prev_price_{shift}'] = df['close_price'].shift(shift)
        df = df.dropna()
        return df

    @staticmethod
    def future_close(df: DataFrame, shift=1) -> DataFrame:
        if shift <= 0:
            raise ValueError("Shift can not be negative or equals zero")

        df = df.copy()
        df[f'future_price_{shift}'] = df['close_price'].shift(-shift)
        df = df.dropna()
        return df

    @staticmethod
    def result_to_bool(df: DataFrame):
        df['TP'] = (df['close'] < df['future']) & (df['close'] < df['predict'])
        df['TN'] = (df['close'] > df['future']) & (df['close'] > df['predict'])
        df['FP'] = (df['close'] > df['future']) & (df['close'] < df['predict'])
        df['FN'] = (df['close'] < df['future']) & (df['close'] > df['predict'])
        return df

    @staticmethod
    @dispatch(list)
    def template_preprocess(data_list: list):
        df = Preprocessor.init_data(data_list)
        return Preprocessor.template_preprocess(df)

    @staticmethod
    @dispatch(DataFrame)
    def template_preprocess(df: DataFrame):
        return Preprocessor.template_preprocess(df, templates=[])

    @staticmethod
    @dispatch(list, list)
    def template_preprocess(data_list: list, templates: list[tuple]):
        df = Preprocessor.init_data(data_list)
        return Preprocessor.template_preprocess(df, templates)

    @staticmethod
    @dispatch(DataFrame, list)
    def template_preprocess(df: DataFrame, templates: list[tuple]):
        for template in templates:
            if template[1] is None:
                df = template[0](df)
            else:
                df = template[0](df, **template[1])

        return df

    @staticmethod
    def preprocess(data_list: list) -> DataFrame:
        df = Preprocessor.init_data(data_list)
        df = Preprocessor.cleared_data(df)
        df = Preprocessor.future_close(df, 1)
        df = Preprocessor.prev_close(df, 1)

        return df
