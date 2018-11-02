from functools import reduce, wraps
from typing import List, Generator, Dict, Callable
from toolz import compose
import logging
import inspect

class ProcessingPipeline(object):
    """A simple callable processing pipeline for imperative execution runtime.

    This class will compose processing functions together to become a callable
    object that takes in the input from the very first processing function and
    returns the output of the last processing function.

    Example:
        This class can be used to create a callable vectorization object which
        will transform a string into a vector and also preserve the preprocessing
        functions for being reused later.
        ```python
        jp = JobPostingCollectionSample()
        vectorization = ProcessingPipeline(
            normalize,
            clean_html,
            clean_str,
            word_tokenize,
            partial(vectorize, embedding_model=w2v)
        )

        vector = vecotrization("Why so serious?")
        ```

     Attributes:
        functions (generator): a series of functions

    """
    def __init__(self, *functions: Callable):
        self.functions = functions

    def __call__(self, input_to_be_processed):
        reversed_for_compose = tuple(reversed(self.functions))
        return compose(*reversed_for_compose)(input_to_be_processed)


class IterablePipeline(object):
    """A simple iterable processing pipeline.

    This class will compose processing functions together to be passed to different stages(training/prediction)
    to assert the same processing procedrues.

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
    preprocessed_generator = pipe(jp)
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

    def __call__(self, source_data_generator: Generator):
        reversed_for_compose = tuple(reversed(self.generators))
        return compose(*reversed_for_compose)(source_data_generator)

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
    if inspect.isgeneratorfunction(func):
        return func
    else:
        @wraps(func)
        def wrapper(*args, **kwargs):
            for item in args[0]:
                if func(item) is not None:
                    yield func(item)
        return wrapper

