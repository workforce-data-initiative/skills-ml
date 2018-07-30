from functools import reduce

def compose(*functions):
    return reduce(lambda f, g: lambda x: g(f(x)), functions, lambda x: x)


class NLPPipeline(object):
    def __init__(self, *functions):
        self.functions = list(functions)

    @property
    def compose(self):
        return reduce(lambda f, g: lambda x: g(f(x)), self.functions, lambda x: x)

    def run(self, generator):
        return list(self.compose(generator))

    @property
    def description(self):
        return [f.__doc__ for f in self.functions]
