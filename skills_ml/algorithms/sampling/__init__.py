"""Generate and store samples of datasets"""

from io import BytesIO


class Sample(object):
    def __init__(self, storage, sample_name):
        self.storage = storage
        self.name = sample_name

    @property
    def base_path(self):
        return self.storage.path

    def __iter__(self):
        fh = BytesIO(self.storage.load(self.name))
        for line in fh:
            yield line

    def __len__(self):
        return sum(1 for item in self)
