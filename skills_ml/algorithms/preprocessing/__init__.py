from functools import reduce
from typing import List, Generator, Dict

class IterablePipeline(object):
    """A simple nlp preprocessing pipeline.

    This class will compose preprocessing functions together to be passed to different stages(training/prediction)
    to assert the same preprocessing procedrues.

    Example:
    ```python
    jp = JobPostingCollectionSample()
    pipe = NLPPipeline(
        partial(fields_joiner, document_schema_fields=['description']),
        html_cleaner,
        sentence_tokenizer,
        str_cleaner,
        word_tokenizer
    )
    result = pipe.run(jp)
    ```

    Attributes:
        functions (generator): a series of generator functions that takes another generator as input

    """
    def __init__(self, *functions: List[Generator]):
        self.functions = functions

    @property
    def compose(self):
        """compose functions

        Returns:
            generator: a generator objet which is not materialized yet
        """
        return reduce(lambda f, g: lambda x: g(f(x)), self.functions, lambda x: x)

    def build(self, source_data_generator: Generator):
        """

        Returns:
            list: a list of itmes after apply all the functions on them
        """
        return ( self.compose(item) for item in source_data_generator)

    @property
    def description(self):
        """pipeline description"""
        return [f.__doc__ for f in self.functions]
