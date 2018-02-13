"""Aggregation functions that can be used with pandas dataframes"""
from collections import Counter


class AggregateFunction(object):
    """Wrap a function with an attribute that indicates the return type name"""
    def __init__(self, returns):
        self.returns = returns

    def __call__(self, function, *params, **kwparams):

        class DecoratedFunction(object):
            def __init__(self, returns, function):
                self.returns = returns
                self.function = function
                self.__name__ = function.__name__
                self.__qualname__ = function.__qualname__
                self.__doc__ = function.__doc__

            def __call__(self, *params, **kwparams):
                return self.function(*params, **kwparams)

        return DecoratedFunction(self.returns, function)


@AggregateFunction(returns='list')
def n_most_common(n, iterable):
    return [mc[0] for mc in Counter(iterable).most_common(n)]


@AggregateFunction(returns='list')
def listy_n_most_common(n, iterable):
    """Expects each item to be iterable, each sub-item to be addable"""
    bc = Counter()
    for i in iterable:
        bc += Counter(i)
    if bc:
        return [mc[0] for mc in bc.most_common(n)]
    else:
        return []
