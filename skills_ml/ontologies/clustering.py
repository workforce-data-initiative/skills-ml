from collections import MutableMapping, KeysView

class KeysViewOnlyKeys(KeysView):
    # Subclass that just changes the representation.
    def __repr__(self):
        return f"{self.__class__.__name__}({list(self)})"


class Clustering(MutableMapping):
    """ A clustering object acting like a dictionary which key is a cluster concept
    and value is a list of entities associated to that cluster.

    Note:
        Python allows a key to be custom object and not necessarily to be string or
        integer as long as it's hashable, but an object key can be difficult to access.
        `key_transform_fn` is to transform an object key to something else, like string
        or integer for easier accessing clustering.

        `value_item_transfrom_fn` is to convert an abstract entity object into something
        else like string, or integer for further computation. It could be a function to
        concat several attributes of the object.

    Example:
        To create a clustering object that we will iterate through a series of concept and
        entity objects, and build the whole thing, we want to extract the concept name
        attribute as the key and make a tuple of entity's identifier and name as the value.
        ```python
        d = Clustering(
                    name="major_group_competencies_name",
                    key_transform_fn=lambda concept: getattr(concept, "name"),
                    value_item_transform_fn=lambda entity: (getattr(entity, "identifier"), getattr(entity, "name")),
            )
     ```

    Args:
        name (str): Name of the clustering
        key_transform_fn (func): the transform function for keys
        value_item_transform_fn (func): the transform function for values

    """
    def __init__(self,
            name,
            key_transform_fn=lambda concept: concept,
            value_item_transform_fn=lambda entity: entity,):
        self.name = name
        self.store = dict()
        self.map_raw_key = dict()
        self.map_raw_value = dict()
        self.key_transform_fn = key_transform_fn
        self.value_item_transform_fn = value_item_transform_fn

    def __setitem__(self, key, value):
        self.store[self.__keytransform__(key)] = [self.value_item_transform_fn(v) for v in value]
        self.map_raw_value[self.__keytransform__(key)] = value
        self.map_raw_key[self.__keytransform__(key)] = key

    def __getitem__(self, key):
        return self.store[key]

    def __delitem__(self, key):
        del self.store[key]
        del self.map_raw_key[key]
        del self.map_raw_value[key]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def __keytransform__(self, key):
        return self.key_transform_fn(key)

    def keys(self):
        return KeysViewOnlyKeys(self)

    def raw_items(self):
        return self.map_raw_value.items()
