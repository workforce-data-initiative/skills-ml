from collections import namedtuple
import json
import logging

import boto
import fiona
import shapely.geometry

from skills_ml.datasets.cbsa_shapefile import download_shapefile
from skills_utils.s3 import split_s3_path

Match = namedtuple('Match', ['index', 'area'])


class S3CachedCBSAFinder(object):
    """Find CBSAs associated with geocode results and save them to S3

    Geocode results are expected in the json format provided by the python 
    `geocoder` module, with a 'bbox'

    The highest-level interface is the 'find_all_cbsas_and_save' method, which
    provides S3 caching. An minimal call looks like:

    cbsa_finder = S3CachedCBSAFinder(s3_conn=..., cache_s3_path='some-bucket/cbsas.json')
    cbsa_finder.find_all_cbsas_and_save({
        "Flushing, NY": { 'bbox': ['southwest': [..., ...], 'northeast': [...,...] }
        "Houston, TX": { 'bbox': ['southwest': [..., ...], 'northeast': [...,...] }
    })

    This usage of 'bbox' is what you can retrieve from a `geocoder` call, such as:
    geocoder.osm('Flushing, NY').json()

    The keys in the resulting cache will be the original search strings.
    
    Warning: The caching is not parallel-safe! It is recommended you should run
    only one copy of `find_all_cbsas_and_save` at a time to avoid overwriting
    the S3 cache file.

    Args: 
        s3_conn (boto.s3.connection) an s3 connection
        cache_s3_path (string) path (including bucket) to the json cache on s3
        shapefile_name (string) local path to a CBSA shapefile to use
            optional, will download TIGER 2015 shapefile if absent
        cache_dir (string) local path to a cache directory to use if the
            shapefile needs to be downloaded
            optional, will use 'tmp' in working directory if absent
    """
    def __init__(
        self,
        s3_conn,
        cache_s3_path,
        shapefile_name=None,
        cache_dir=None
    ):
        self.s3_conn = s3_conn
        self.cache_s3_path = cache_s3_path
        self.shapes = []
        self.properties = []
        self.cache = None
        self.cache_original_size = 0
        self.shapefile_name = shapefile_name or download_shapefile(cache_dir or 'tmp')

    @property
    def _key(self):
        bucket_name, path = split_s3_path(self.cache_s3_path)
        return boto.s3.key.Key(
            bucket=self.s3_conn.get_bucket(bucket_name),
            name=path
        )

    def _load_cache(self):
        """Load the result cache into memory"""
        try:
            cache_json = self._key.get_contents_as_string().decode('utf-8')
            self.cache = json.loads(cache_json)
            if not self.cache:
                self.cache = {}
            self.cache_original_size = len(cache_json)
        except boto.exception.S3ResponseError as e:
            logging.warning(
                'CBSA finder cachefile load failed with exception %s,' +
                'will overwrite', e
            )
            self.cache = {}

    def _load_shapefile(self):
        """Load the CBSA Shapefile into memory"""
        with fiona.collection(self.shapefile_name) as input:
            for row in input:
                self.shapes.append(shapely.geometry.shape(row['geometry']))
                self.properties.append(row['properties'])

        try:
            cache_json = self._key.get_contents_as_string().decode('utf-8')
            self.cache = json.loads(cache_json)
            if not self.cache:
                self.cache = {}
            self.cache_original_size = len(cache_json)
        except boto.exception.S3ResponseError as e:
            logging.warning(
                'CBSA finder cachefile load failed with exception %s,' +
                'will overwrite', e
            )
            self.cache = {}

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
        if 'bbox' not in geocode_result:
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
        """Find CBSAs from geocode results and save the results to S3

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
        """Save the cbsa finding cache to S3"""
        cache_json = json.dumps(self.cache)
        if len(cache_json) >= self.cache_original_size:
            new_cache_json = json.dumps(self.cache)
            self._key.set_contents_from_string(new_cache_json)
            logging.info(
                'Successfully saved cbsa finding cache to %s',
                self.cache_s3_path
            )
        else:
            logging.error(
                'New cache size: %s smaller than existing cache size: %s, aborting',
                len(cache_json),
                self.cache_original_size
            )

    @property
    def all_cached_cbsa_results(self):
        """Return the contents of the cache

        Returns: (dict) search strings mapping to their (tuple) results
        """
        if not self.cache:
            self._load_cache()
        return self.cache
