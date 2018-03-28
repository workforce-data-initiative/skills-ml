"""Use the Census CBSA shapefile"""
import requests
import os
import zipfile
import filelock

DATASET_URL = 'http://www2.census.gov/geo/tiger/TIGER2015/CBSA/tl_2015_us_cbsa.zip'


def download_shapefile(cache_dir):
    """Download Tiger 2015 CBSA Shapefile

    Downloads the zip archive and unzips the contents

    Args:
        cache_dir (string) local path to download files to

    Returns: (string) Path to the extracted shapefile
    """
    lock = filelock.FileLock(os.path.join(cache_dir, 'cbsa_shp_dl.lock'))
    with lock.acquire(timeout=1000):
        write_path = os.path.join(cache_dir, 'tl_2015_us_cbsa.zip')
        if not os.path.isfile(write_path):
            response = requests.get(DATASET_URL, stream=True)
            with open(write_path, 'wb') as fd:
                for chunk in response.iter_content(chunk_size=128):
                    fd.write(chunk)
        zip_ref = zipfile.ZipFile(write_path, 'r')
        zip_ref.extractall(cache_dir)
        zip_ref.close()
    return os.path.join(cache_dir, 'tl_2015_us_cbsa.shp')
