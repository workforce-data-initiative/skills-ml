"""Common utilities"""

def safe_get(dct, *keys):
	"""Extract value from nested dictionary
	Args:
		dct (dict): dictionary one want to extrat value from
		*keys (string|list|tuple): keys to exatract value

	Returns:
		value
	"""
	for key in keys:
		try:
			dct = dct[key]
		except:
			return None
	return dct

