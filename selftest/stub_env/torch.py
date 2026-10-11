"""Minimal torch stub: only what dapt_lib touches, nothing else."""

bfloat16 = "bfloat16"
float16 = "float16"


class _Tensor:
    def __init__(self, n=1000, rg=True):
        self._n = n
        self.requires_grad = rg
        self.grad = None

    def numel(self):
        return self._n

    def norm(self):
        return _Tensor(1)

    def item(self):
        return 0.5

    def all(self):
        return True

    def clone(self):
        return _Tensor(self._n, self.requires_grad)

    def __eq__(self, other):
        return _Tensor()

    def __deepcopy__(self, memo):
        return _Tensor(self._n, self.requires_grad)

    def __repr__(self):
        return "_Tensor(%d, rg=%s)" % (self._n, self.requires_grad)


class _Cuda:
    _peak = 12.3e9

    def empty_cache(self):
        pass

    def reset_peak_memory_stats(self):
        _Cuda._peak = 12.3e9

    def max_memory_allocated(self):
        return _Cuda._peak


cuda = _Cuda()


class no_grad:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False
