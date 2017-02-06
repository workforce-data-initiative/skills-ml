import httpretty

from datasets.ua_cbsa import ua_cbsa, URL

RESPONSE = '''UA,UANAME,CBSA,MNAME,MEMI,POPPT,HUPT,AREAPT,AREALANDPT,UAPOP,UAHU,UAAREA,UAAREALAND,MPOP,MHU,MAREA,MAREALAND,UAPOPPCT,UAHUPCT,UAAREAPCT,UAAREALANDPCT,MPOPPCT,MHUPCT,MAREAPCT,MAREALANDPCT
00037,"Abbeville, LA Urban Cluster",10020,"Abbeville, LA Micro Area",2,19268,8216,28218638,27918141,19824,8460,29523368,29222871,57999,25235,3993941933,3038572441,97.2,97.12,95.58,95.54,33.22,32.56,.71,.92
00037,"Abbeville, LA Urban Cluster",35340,"New Iberia, LA Micro Area",2,556,244,1304730,1304730,19824,8460,29523368,29222871,73240,29698,2669055888,1486940445,2.8,2.88,4.42,4.46,.76,.82,.05,.09
00064,"Abbeville, SC Urban Cluster",99999,"Not in a metro/micro area",,5243,2578,11334983,11315197,5243,2578,11334983,11315197,,,,,100,100,100,100,,,,
00091,"Abbotsford, WI Urban Cluster",48140,"Wausau, WI Metro Area",1,1103,428,2102170,2102170,3966,1616,5376662,5363441,134063,57734,4082627087,4001488029,27.81,26.49,39.1,39.19,.82,.74,.05,.05
00091,"Abbotsford, WI Urban Cluster",99999,"Not in a metro/micro area",,2863,1188,3274492,3261271,3966,1616,5376662,5363441,,,,,72.19,73.51,60.9,60.81,,,,
00118,"Aberdeen, MS Urban Cluster",99999,"Not in a metro/micro area",,4666,2050,7469348,7416616,4666,2050,7469348,7416616,,,,,100,100,100,100,,,,'''


@httpretty.activate
def test_ua_cbsa():
    httpretty.register_uri(
        httpretty.GET,
        URL,
        body=RESPONSE,
        content_type='text/csv'
    )

    results = ua_cbsa.__wrapped__()
    assert results == {
        '00037': ['10020', '35340'],
        '00091': ['48140']
    }
