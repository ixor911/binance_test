class Counter:
    def __init__(self, end: int, start: int = 0):
        self.end = end
        self.value = start

    def count(self):
        self.value += 1

    def show(self):
        print(f"[{self.value}/{self.end}]")

    def next(self):
        self.count()
        self.show()
