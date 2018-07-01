"""Download ONET files from their site"""
import requests


def download_onet(source_file):
    url_prefix = 'http://www.onetcenter.org/dl_files/database/db_22_3_text'
    url = f'{url_prefix}/{source_file}.txt'
    response = requests.get(url)
    return response.text
