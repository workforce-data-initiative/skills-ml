from algorithms.representativeness_calculators.geo_occupation import \
    GeoOccupationRepresentativenessCalculator


class FakeNormalizer(object):
    def normalize_job_title(self, job_title):
        if job_title == 'Cupcake Ninja':
            return '11-1011.00'
        else:
            return '11-1013.00'


class FakeCBSAQuerier(object):
    def query(self, job_listing):
        if job_listing['id'] == 1:
            return ('123', 'Chicago, IL')
        else:
            return ('456', 'Boston, MA')

sample_jobs = [
    {'id': 1, 'onet_soc_code': '11-1012.00', 'title': 'Cupcake Ninja'},
    {'id': 2, 'title': 'Cupcake Ninja'},
    {'id': 3, 'title': 'React Ninja'},
    {'id': 4, 'title': 'React Ninja'},
]


def test_distribution_calculator():
    calculator = GeoOccupationRepresentativenessCalculator(
        geo_querier=FakeCBSAQuerier(),
        normalizer=FakeNormalizer()
    )
    distribution = calculator.dataset_distribution(sample_jobs)
    assert distribution == {
        ('456', '11-1013.00'): 2,
        ('456', '11-1011.00'): 1,
        ('123', '11-1012.00'): 1
    }


def test_distribution_calculator_no_normalize():
    calculator = GeoOccupationRepresentativenessCalculator(
        geo_querier=FakeCBSAQuerier(),
    )
    distribution = calculator.dataset_distribution(sample_jobs)
    assert distribution == {
        ('123', '11-1012.00'): 1
    }
