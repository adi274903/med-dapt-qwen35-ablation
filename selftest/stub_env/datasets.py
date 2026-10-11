"""Fake streaming-dataset layer: enough of `datasets` to run build_data."""
import itertools


class _Stream:
    """Iterable, like a streaming dataset. `finite=True` mimics `.take(n)`."""

    def __init__(self, items, name="s", finite=False):
        self.items = list(items)
        self.name = name
        self.finite = finite
        self.column_names = ["contents", "title", "content", "input", "output",
                             "description", "text", "question", "answer"]

    def map(self, fn, **kw):
        return _Stream([fn(x) for x in self.items], self.name, self.finite)

    def filter(self, fn):
        return _Stream([x for x in self.items if fn(x)], self.name, self.finite)

    def shuffle(self, **kw):
        return self

    def skip(self, n):
        return self

    def take(self, n):
        out = []
        while len(out) < n:
            out.extend(self.items)
        return _Stream(out[:n], self.name, finite=True)

    def __iter__(self):
        if self.finite:
            yield from self.items
            return
        for i in itertools.count():
            yield self.items[i % len(self.items)]



class Dataset:
    @staticmethod
    def from_list(rows):
        class _D:
            def __init__(self, r):
                self.r = r

            def __len__(self):
                return len(self.r)

            def __iter__(self):
                return iter(self.r)

        return _D(rows)


class IterableDataset:
    @staticmethod
    def from_generator(gen, **kw):
        class _I:
            def __init__(self, g):
                self.g = g

            def __iter__(self):
                return iter(self.g)

        return _I(gen)


def load_dataset(path, cfg=None, split=None, streaming=False, **kw):
    rows = [{"contents": "%s row %d" % (path, i), "text": "%s row %d" % (path, i),
             "question": "q%d" % i, "answer": "a%d" % i, "title": "t", "content": "c",
             "input": "i", "output": "o", "description": "d"} for i in range(600)]
    return _Stream(rows, name=path)


def interleave_datasets(streams, probabilities=None, seed=None, stopping_strategy=None):
    return _Stream([x for s in streams for x in s.items[:5]],
                   name="interleaved(%d)" % len(streams))
