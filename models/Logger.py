import datetime
import json
from .Model import Model
from pandas import DataFrame
import os
import time


class Logger:
    @staticmethod
    def templates_to_strs(templates: list):
        return [template[0].__name__ for template in templates]

    @staticmethod
    def count_predicts(df: DataFrame):
        result = {
            "all": len(df),
            "TP": len(df.loc[df['TP']]),
            "TN": len(df.loc[df['TN']]),
            "FP": len(df.loc[df['FP']]),
            "FN": len(df.loc[df['FN']]),
        }
        result['good'] = result.get('TP') + result.get("TN")
        result['bad'] = result.get('FP') + result.get("FN")

        return result

    @staticmethod
    def create_dir(fp: str):
        counter = 0
        while True:
            try:
                os.mkdir(f"{fp}")
                return
            except:
                counter += 1
                fp += "(1)"

    @staticmethod
    def save_result(name: str, result: dict, params: dict, model: Model, templates: list, df: DataFrame):
        Logger.create_dir(f"results/{name}")

        json.dump(result, open(f"results/{name}/result.json", 'w'), indent=4)
        json.dump(params, open(f"results/{name}/params.json", 'w'), indent=4)
        json.dump({"templates": templates}, open(f"results/{name}/templates.json", 'w'), indent=4)
        model.save(f"results/{name}", "model")
        # df.to_excel(f"results/{name}/best_df.xlsx")

    @staticmethod
    def check_result(fp):
        try:
            files = os.listdir(fp)
            need_files = [name in files for name in ['model.pkl', 'params.json', 'result.json', 'templates.json']]
            if False not in need_files:
                return True
        except FileNotFoundError:
            print(FileNotFoundError.__name__)
            return False

    @staticmethod
    def load_result(fp: str):
        if not Logger.check_result(fp):
            return None

        model = Model()
        model.load(f"{fp}/model.pkl")
        params = json.load(open(f"{fp}/params.json", 'r'))
        result = json.load(open(f"{fp}/result.json", 'r'))
        templates = json.load(open(f"{fp}/templates.json", 'r'))

        return {
            "fp": fp,
            "model": model,
            "result": result,
            "params": params,
            "templates": templates
        }

    @staticmethod
    def load_best_result(feature: str = "good_max"):
        results = os.listdir("results/")
        best_result = None
        for result_name in results:
            if not Logger.check_result(f"results/{result_name}"):
                continue

            result = Logger.load_result(f"results/{result_name}")

            if best_result is None or result.get('result').get(feature) > best_result.get('result').get(feature):
                best_result = result

        return best_result

