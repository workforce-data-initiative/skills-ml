from smart_open import smart_open


class Sample(object):
    def __init__(self, base_path, sample_name):
        self.base_path = base_path
        self.name = sample_name
        self.full_path = '/'.join([self.base_path, self.name])

    def __iter__(self):
        lines = []
        with smart_open(self.full_path) as f:
            lines = [line for line in f]
        for line in lines:
            yield line
