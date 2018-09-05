from functools import reduce, wraps
from typing import List, Generator, Dict, Callable
import logging

class IterablePipeline(object):
    """A simple iterable preprocessing pipeline.

    This class will compose preprocessing functions together to be passed to different stages(training/prediction)
    to assert the same preprocessing procedrues.

    Example:
    ```python
    jp = JobPostingCollectionSample()
    pipe = IterablePipeline(
        partial(fields_join, document_schema_fields=['description']),
        clean_html,
        sentence_tokenize,
        clean_str,
        word_tokenize
    )
    preprocessed_generator = pipe.build(jp)
    ```

    Attributes:
        functions (generator): a series of generator functions that takes another generator as input

    """
    def __init__(self, *functions: Callable):
        self.functions = functions
        self._generators = [func2gen(f) for f in self.functions]

    @property
    def generators(self):
        return self._generators

    @generators.setter
    def generators(self, new_generators):
        self._generators = new_generators

    @property
    def _compose(self):
        """compose functions

        Returns:
            function: a function object which is not materialized yet
        """
        return reduce(lambda f, g: lambda x: g(f(x)), self._generators, lambda x: x)

    def build(self, source_data_generator: Generator):
        """

        Returns:
            generator: a generator object of itmes after apply all the functions on them
        """
        return self._compose(source_data_generator)

    @property
    def description(self):
        """pipeline description"""
        return [f.__doc__ for f in self.functions]


def func2gen(func: Callable) -> Callable:
    """A wrapper that change a document-transforming function that takes only one document the input
    into a function that takes a generator/iterator as the input. When it instantiates, it will become
    a generator.

    Example:
        @func2gen
        def do_something(doc):
            return do_something_to_the_doc(doc)

    Args:
        func (function): a function only take one document as the first argument input.

    Returns:
        func (function): a function that takes a generator as the first argument input.

    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        for item in args[0]:
            if item is not None:
                yield func(item)
    return wrapper

