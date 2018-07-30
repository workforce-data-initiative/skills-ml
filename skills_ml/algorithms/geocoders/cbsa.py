"""Given geocode results, find matching Core-Based Statistical Areas."""
from collections import namedtuple
import json
import logging

import boto
import s3fs
import shapely.geometry
import fiona

from skills_ml.datasets.cbsa_shapefile import download_shapefile
from skills_ml.storage import PersistedJSONDict

Match = namedtuple('Match', ['index', 'area'])


class CachedCBSAFinder(object):
    """Find CBSAs associated with geocode results and save them to the specified storage

    Geocode results are expected in the json format provided by the python
    `geocoder` module, with a 'bbox'

    The highest-level interface is the 'find_all_cbsas_and_save' method, which
    provides storage caching. A minimal call looks like

    ```python
    cache_storage = S3Store('some-bucket')
    cache_fname = 'cbsas.json'
    cbsa_finder = CachedCBSAFinder(cache_storage=cache_storage, cache_fname=cache_fname)
    cbsa_finder.find_all_cbsas_and_save({
        "Flushing, NY": { 'bbox': ['southwest': [..., ...], 'northeast': [...,...] }
        "Houston, TX": { 'bbox': ['southwest': [..., ...], 'northeast': [...,...] }
    })

    # This usage of 'bbox' is what you can retrieve from a `geocoder` call, such as:
    geocoder.osm('Flushing, NY').json()
    ```

    The keys in the resulting cache will be the original search strings.

    Warning: The caching is not parallel-safe! It is recommended you should run
    only one copy of `find_all_cbsas_and_save` at a time to avoid overwriting
    the cache file.

    Args:
        cache_storage (object) FSStore() or S3Store object to store the cache
        cache_fname (string) cache file name
        shapefile_name (string) local path to a CBSA shapefile to use
            optional, will download TIGER 2015 shapefile if absent
        cache_dir (string) local path to a cache directory to use if the
            shapefile needs to be downloaded
            optional, will use 'tmp' in working directory if absent
    """
    def __init__(
        self,
        cache_storage,
        cache_fname,
        shapefile_name=None,
        cache_dir=None
    ):
        self.cache_storage = cache_storage
        self.cache_fname = cache_fname
        self.shapes = []
        self.properties = []
        self.cache = PersistedJSONDict(self.cache_storage, self.cache_fname)
        self.cache_dir = cache_dir
        self.shapefile_name = shapefile_name

    def _load_shapefile(self):
        """Load the CBSA Shapefile into memory"""
        if not self.shapefile_name:
            self.shapefile_name = download_shapefile(self.cache_dir or 'tmp')
        with fiona.collection(self.shapefile_name) as input:
            for row in input:
                self.shapes.append(shapely.geometry.shape(row['geometry']))
                self.properties.append(row['properties'])

    def query(self, geocode_result):
        """Find the geographically closest CBSA to the given geocode result

        Args:
            geocode_result (dict) The result of a geocoding call.
                Expected to contain a bounding box under 'bbox'

        Returns: (tuple) the FIPS and name of the CBSA with the most extensive
            intersection with the given bounding box,
            or None if no CBSAs intersect with it
        """
        if not self.shapes or not self.properties:
            self._load_shapefile()
        if not geocode_result or 'bbox' not in geocode_result:
            logging.warning('Geocode result failed: %s', geocode_result)
            return None
        box = shapely.geometry.box(
            minx=geocode_result['bbox']['southwest'][1],
            miny=geocode_result['bbox']['southwest'][0],
            maxx=geocode_result['bbox']['northeast'][1],
            maxy=geocode_result['bbox']['northeast'][0]
        )
        matches = []
        for shape_index, shape in enumerate(self.shapes):
            if box.intersects(shape):
                matches.append(
                    Match(area=box.intersection(shape).area, index=shape_index)
                )
        if len(matches) > 1:
            best_match = sorted(
                matches,
                key=lambda match: match.area,
                reverse=True
            )[0]
            logging.warning(
                'More than one match found for %s All matches: %s Picking %s',
                geocode_result,
                [self.properties[match.index]['NAMELSAD'] for match in matches],
                self.properties[best_match.index]['NAMELSAD']
            )
        elif len(matches) == 0:
            logging.warning('Unable to find a CBSA match for %s', geocode_result)
            return None
        else:
            best_match = matches[0]

        properties = self.properties[best_match.index]
        return (properties['CBSAFP'], properties['NAMELSAD'])

    def find_all_cbsas_and_save(self, geocode_results):
        """Find CBSAs from geocode results and save the results to storage

        Args:
            geocode_results (dict) Search strings mapping to geocode results
        """
        try:
            for search_string, geocode_result in geocode_results.items():
                self.cache[search_string] = self.query(geocode_result)
            self.save()
        except Exception as e:
            logging.warning('Quitting cbsa finding due to %s', e)


    def save(self):
        """Save the cbsa finding cache to the specified storage"""
        self.cache.save()

    @property
    def all_cached_cbsa_results(self):
        """Return the contents of the cache

        Returns: (dict) search strings mapping to their (tuple) results
        """
        return self.cache
