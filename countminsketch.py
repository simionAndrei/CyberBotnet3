import numpy as np
import mmh3


class CountMinSketch:
    def __init__(self, width: int, depth: int, seeds=None):
        self.width = width
        self.depth = depth
        self.table = np.zeros([depth, width])
        self.seed = seeds if seeds is not None else np.random.randint(width, size=depth)

    def add(self, key: str):

        for i in range(self.depth):
            j = mmh3.hash(key, self.seed[i]) % self.width
            self.table[i, j] += 1

    def __getitem__(self, key: str):
        min_est = self.width
        for i in range(self.depth):
            j = mmh3.hash(key, self.seed[i]) % self.width

            est = self.table[i, j]
            if est < min_est:
                min_est = est

        return min_est


if __name__ == "__main__":
    c = CountMinSketch(4, 4)
    assert c["k"] == 0
    c.add("k")
    assert c["k"] > 0
    assert c["j"] == 0
