import httpretty

from datasets.cousub_ua import cousub_ua, URL

RESPONSE = '''UA,UANAME,STATE,COUNTY,COUSUB,GEOID,CSNAME,POPPT,HUPT,AREAPT,AREALANDPT,UAPOP,UAHU,UAAREA,UAAREALAND,CSPOP,CSHU,CSAREA,CSAREALAND,UAPOPPCT,UAHUPCT,UAAREAPCT,UAAREALANDPCT,CSPOPPCT,CSHUPCT,CSAREAPCT,CSAREALANDPCT
00334,"Abingdon, IL Urban Cluster",17,095,37361,1709537361,"Indian Point township (Knox County)",1132,505,1339458,1339458,3389,1483,3731303,3731303,1554,716,93483992,93455686,33.4,34.05,35.9,35.9,72.84,70.53,1.43,1.43
00388,"Ada, OH Urban Cluster",39,065,43162,3906543162,"Liberty township (Hardin County)",5945,1906,4769036,4769036,5945,1906,4769036,4769036,7712,2615,92826662,92822103,100,100,100,100,77.09,72.89,5.14,5.14
'''


@httpretty.activate
def test_cousub_ua():
    httpretty.register_uri(
        httpretty.GET,
        URL,
        body=RESPONSE,
        content_type='text/csv'
    )

    results = cousub_ua.__wrapped__(lambda s: s.lower())
    assert results == {
        'IL': {'indian point': '00334'},
        'OH': {'liberty': '00388'},
    }
