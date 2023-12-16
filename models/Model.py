from sklearn.ensemble import RandomForestRegressor
from pandas import DataFrame
import pickle
from multipledispatch import dispatch


class Model:
    @dispatch()
    def __init__(self):
        self.model = RandomForestRegressor()

    @dispatch(dict)
    def __init__(self, params: dict):
        self.model = RandomForestRegressor(**params)

    def fit(self, x, y):
        self.model.fit(x, y)  # тут есть веса, с которыми тоже можно поиграться

    def predict(self, x) -> list:
        return self.model.predict(x)

    def test(self, x_test, y_test) -> DataFrame:
        df_result = DataFrame(data={
            "close": x_test['close_price'],
            "future": y_test,
            "predict": self.predict(x_test)
        })
        return df_result

    def params(self):
        return self.model.get_params()

    def save(self, fp: str, fn: str):
        with open(f"{fp}/{fn}.pkl", 'wb') as file:
            pickle.dump(self.model, file)

    def load(self, fp):
        with open(fp, 'rb') as file:
            self.model = pickle.load(file)

