import os
import threading

from binance.spot import Spot
from models.TimeConvertor import Convertor
import json
from models.Counter import Counter


class DataProvider:
    @staticmethod
    def retry_till_result(func, **kwargs):
        while True:
            try:
                return func(**kwargs)
            except:
                print("Retrying...")

    @staticmethod
    def get_interval_time(symbol: str, interval: str):
        data = DataProvider.retry_till_result(
            Spot().klines,
            symbol=symbol,
            interval=interval,
            limit=2
        )
        return (data[1][0] - data[0][0]) * 1000

    @staticmethod
    def thread_get_data(symbol: str, interval: str, start_time: int, buffer: list, counter: Counter):
        data = DataProvider.retry_till_result(
            Spot().klines,
            symbol=symbol,
            interval=interval,
            limit=1000,
            startTime=start_time
        )
        buffer.append(data)
        counter.next()

    @staticmethod
    def create_data(symbol: str, interval: str, years: int):
        print(f"\ncreate_data, {symbol}_{interval}_{years}y:")

        end_time = Spot().time().get('serverTime')
        start_time = end_time - Convertor.year * years
        interval_time = DataProvider.get_interval_time(symbol, interval)

        counter = Counter(int((end_time - start_time) / interval_time) + 1)

        full = []
        while start_time < end_time:
            data = DataProvider.retry_till_result(
                Spot().klines,
                symbol=symbol,
                interval=interval,
                limit=1000,
                startTime=start_time
            )
            full += data

            start_time += interval_time
            counter.next()

        json.dump({"data": full}, open(f"data/{symbol}_{interval}_{years}y", "w"), indent=4)

    @staticmethod
    def create_data_async(symbol: str, interval: str, years: int):
        print(f"\ncreate_data, {symbol}_{interval}_{years}y:")

        end_time = Spot().time().get('serverTime')
        start_time = end_time - Convertor.year * years
        interval_time = DataProvider.get_interval_time(symbol, interval)

        counter = Counter(int((end_time - start_time) / interval_time) + 1)

        buffer = []
        threads = []
        while start_time < end_time:
            thread = threading.Thread(
                target=DataProvider.thread_get_data,
                args=[symbol, interval, start_time, buffer, counter]
            )
            thread.daemon = True
            thread.start()
            threads.append(thread)

            start_time += interval_time

        for tread in threads:
            tread.join()

        sorted(buffer, key=lambda x: x[0][0])

        full = []
        for result in buffer:
            full += result

        if not os.path.exists(f"data/{symbol}_{years}y"):
            os.makedirs(f"data/{symbol}_{years}y")

        json.dump({"data": full}, open(f"data/{symbol}_{years}y/{interval}.json", "w"), indent=4)

    @staticmethod
    def load_data(fp: str):
        try:
            data = json.load(open(fp, 'r'))
            return data.get('data')
        except FileNotFoundError:
            print("File not found")


























