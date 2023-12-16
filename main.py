from models import *
import datetime


data_list = DataProvider.load_data("data/BTCUSDT_5y/1d.json")
df = Preprocessor.template_preprocess(
    data_list,
    [
        (Preprocessor.cleared_data, None),
        (Preprocessor.future_close, {"shift": 1}),
    ]
)

templates = [
    (Preprocessor.prev_close, {"shift": 1}),
    (Preprocessor.prev_close, {"shift": 2}),
    (Preprocessor.prev_close, {"shift": 3}),
]

model_params = {
    "criterion": ["squared_error"],
    "min_samples_leaf": range(1, 3),
    "min_samples_split": range(2, 4),
    "n_estimators": range(50, 151, 50)
}

# Tester.combo_test(df, model_params, templates)
Tester.combo_test_async(df, model_params, templates)

# print(Logger.load_best_result())
# print(Logger.load_best_result().get('result'))






