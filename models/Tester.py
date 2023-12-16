import datetime
import threading

from pandas import DataFrame, Series
import itertools

from . import Counter
from .Logger import Logger
from .Model import Model
from .Preprocessor import Preprocessor
import copy
from multipledispatch import dispatch


class Tester:
    @staticmethod
    def get_template_combos(templates_list: list) -> list:
        combinations = []
        for r in range(1, len(templates_list) + 1):
            combinations.extend(itertools.combinations(templates_list, r))
        return combinations

    @staticmethod
    def get_params_combos(params_list: dict) -> list:
        keys, values = zip(*params_list.items())
        permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
        return permutations_dicts

    @staticmethod
    @dispatch(dict, list)
    def get_full_combos(params: dict, templates: list) -> list:
        params_combo = Tester.get_params_combos(params)
        templates_combo = Tester.get_template_combos(templates)

        return Tester.get_full_combos(params_combo, templates_combo)

    @staticmethod
    @dispatch(list, list)
    def get_full_combos(params_combos: list, templates_combo: list):
        full_combo = []
        for param in params_combos:
            for template in templates_combo:
                full_combo.append((param, template))

        return full_combo

    @staticmethod
    @dispatch(DataFrame, Model, amount=int)
    def test(df, model, amount=10) -> dict:
        x_train, y_train, x_test, y_test = Preprocessor.train_test_split(df)
        return Tester.test(x_train, y_train, x_test, y_test, model, amount=amount)

    @staticmethod
    @dispatch(DataFrame, Series, DataFrame, Series, Model, amount=int)
    def test(x_train, y_train, x_test, y_test, model, amount=10) -> dict:
        results = []
        best_df = None
        best_model = None
        best_score = 0

        for i in range(amount):
            model.fit(x_train, y_train)
            df_result = model.test(x_test, y_test)
            df_result = Preprocessor.result_to_bool(df_result)

            counted = Logger.count_predicts(df_result)
            if counted.get("TP") + counted.get("TN") > best_score:
                best_model = copy.deepcopy(model)
                best_score = counted.get("TP") + counted.get("TN")
                best_df = df_result

            results.append(counted)

        result = {
            "best_df": best_df,
            "best_model": best_model,
            "all": results[0].get('all'),
            "TP_avg": sum([res.get('TP') for res in results]) / len(results),
            "TP_max": max([res.get('TP') for res in results]),
            "TN_avg": sum([res.get('TN') for res in results]) / len(results),
            "TN_max": max([res.get('TN') for res in results]),
            "FP_avg": sum([res.get('FP') for res in results]) / len(results),
            "FP_max": max([res.get('FP') for res in results]),
            "FN_avg": sum([res.get('FN') for res in results]) / len(results),
            "FN_max": max([res.get('FN') for res in results]),
        }
        result['good_avg'] = result.get('TP_avg') + result.get("TN_avg")
        result['good_max'] = result.get('TP_max') + result.get("TN_max")
        result['bad_avg'] = result.get('FP_avg') + result.get("FN_avg")
        result['bad_max'] = result.get('FP_max') + result.get("FN_max")

        return result

    @staticmethod
    def combo_test_thread(df: DataFrame, combination: tuple, counter: Counter = None):
        dt = datetime.datetime.now()

        params = combination[0]
        templates = combination[1]

        model = Model(params)

        combo_df = df.copy()

        result = Tester.test(combo_df, model, amount=5)
        best_df = result.pop('best_df')
        best_model = result.pop('best_model')
        templates_names = Logger.templates_to_strs(templates)
        name = f"{dt.year}_{dt.month}_{dt.day}__{dt.hour}_{dt.minute}_{dt.second}_{dt.microsecond}"

        Logger.save_result(
            name=name,
            result=result,
            params=params,
            model=best_model,
            templates=templates_names,
            df=best_df
        )

        if counter is not None:
            counter.next()

    @staticmethod
    def combo_test(df: DataFrame, model_params: dict, templates: list):
        full_combo = Tester.get_full_combos(model_params, templates)
        counter = Counter(end=len(full_combo))

        for combination in full_combo:
            Tester.combo_test_thread(df, combination, counter)

    @staticmethod
    def combo_test_async(df: DataFrame, model_params: dict, templates: list):
        full_combo = Tester.get_full_combos(model_params, templates)
        counter = Counter(end=len(full_combo))

        threads = []
        for combination in full_combo:
            thread = threading.Thread(
                target=Tester.combo_test_thread,
                args=[df, combination, counter]
            )
            thread.daemon = True
            thread.start()
            threads.append(thread)

        for tread in threads:
            tread.join()
