import evaluation.query as query

import random

from tests.utils import makeNamedTemporaryCSV


def test_NormalizerResponse():
    random.seed(266)
    class TestNormalizerResponse(query.NormalizerResponse):
        def normalize(self, job_title):
            return [
                {'title': '{} {}'.format(job_title, i), 'relevance_score': 0.5}
                for i in range(0,5)
            ]

        def _good_response(self, response):
            return True

    content = [
        ['Cupcake Ninja', 'a baker', '1234'],
        ['Oyster Floater', 'a person that floats oysters', '2345'],
    ]

    with makeNamedTemporaryCSV(content, '\t') as csvname:
        evaluator = TestNormalizerResponse(
            name='test normalizer',
            access=csvname
        )
        result = [
            ranked_row
            for response in evaluator
            for ranked_row in evaluator.ranked_rows(response)
        ]
        assert result == [
            ('Cupcake Ninja', 'a baker', 'Cupcake Ninja 1', 1),
            ('Cupcake Ninja', 'a baker', 'Cupcake Ninja 2', 2),
            ('Cupcake Ninja', 'a baker', 'Cupcake Ninja 0', 0),
            ('Oyster Floater', 'a person that floats oysters', 'Oyster Floater 2', 2),
            ('Oyster Floater', 'a person that floats oysters', 'Oyster Floater 0', 0),
            ('Oyster Floater', 'a person that floats oysters', 'Oyster Floater 1', 1)
        ]
