import httpretty

from datasets.nber_county_cbsa import cbsa_lookup, URL

CBSA_RESPONSE = '''"countyname","state","ssacounty","fipscounty","cbsa","cbsaname","ssast","fipst"
"AUTAUGA","AL","01000","01001","33860","Montgomery, AL","01","01"'''.encode('latin-1')


@httpretty.activate
def test_cbsa_lookup():
    httpretty.register_uri(
        httpretty.GET,
        URL,
        body=CBSA_RESPONSE,
        content_type='text/csv'
    )

    results = cbsa_lookup.__wrapped__()
    assert results == {'AL': {'001': ('33860', 'Montgomery, AL')}}
