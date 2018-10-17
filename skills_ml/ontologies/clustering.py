from collections import MutableMapping, KeysView

class KeysViewOnlyKeys(KeysView):
    # Subclass that just changes the representation.
    def __repr__(self):
        return f"{self.__class__.__name__}({list(self)})"


class Clustering(MutableMapping):
    """ A clustering object acting like a dictionary which key is a cluster concept
    and value is a list of entities associated to that cluster

    Args:
        name (str): Name of the clustering
        key_transform_fn (func): the transform function for keys
        value_transform_fn (func): the transform function for values

    """
    def __init__(self,
            name,
            key_transform_fn=lambda concept: concept,
            value_transform_fn=lambda entity: entity,):
        self.name = name
        self.store = dict()
        self.map_original_key = dict()
        self.key_transform_fn = key_transform_fn
        self.value_transform_fn = value_transform_fn

    def __setitem__(self, key, value):
        self.store[self.__keytransform__(key)] = value
        self.map_original_key[self.__keytransform__(key)] = key

    def __getitem__(self, key):
        return self.store[key]

    def __delitem__(self, key):
        del self.store[key]
        del self.map_original_key[key]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def __keytransform__(self, key):
        return self.key_transform_fn(key)

    def keys(self):
        return KeysViewOnlyKeys(self)

