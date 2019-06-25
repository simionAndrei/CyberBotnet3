import numpy as np
import mmh3
import pandas as pd
import math
from operator import itemgetter

e = 2.718281828


class CountMinSketch:
    def __init__(self, width: int, depth: int, seeds=None):
        self.width = width
        self.depth = depth
        self.__table = np.zeros([depth, width])
        self.__seed = seeds if seeds is not None else np.random.randint(width, size=depth)

    def add(self, key: str):

        for i in range(self.depth):
            j = mmh3.hash(key, self.__seed[i]) % self.width
            self.__table[i, j] += 1

    def add_all(self, keys):
        for key in keys:
            self.add(key)

    def __getitem__(self, key: str):
        min_est = self.width
        for i in range(self.depth):
            j = mmh3.hash(key, self.__seed[i]) % self.width

            est = self.__table[i, j]
            if est < min_est:
                min_est = est

        return min_est

    def extract_top(self, keys):
        results = []
        for key in keys:
            results.append([key, self[key]])

        results.sort(key=itemgetter(1), reverse=True)

        return results

    @staticmethod
    def init_by_error(epsilon, delta, seeds=None):
        width = round(e / epsilon)
        depth = round(math.log(1 / delta))

        return CountMinSketch(width, depth, seeds)


if __name__ == "__main__":
    c = CountMinSketch(4, 4)
    assert c["k"] == 0
    c.add("k")
    assert c["k"] > 0
    assert c["j"] == 0

    c.add_all(["j", "d"])

    assert c["j"] > 0
    assert c["d"] > 0
