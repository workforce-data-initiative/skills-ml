import httpretty

from datasets.negative_dict import negative_dict, PLACEURL, STATEURL

STATERESPONSE = """id,name,abbreviation,country,type,sort,status,occupied,notes,fips_state,assoc_press,standard_federal_region,census_region,census_region_name,census_division,census_division_name,circuit_court
"1","Alabama","AL","USA","state","10","current","occupied","","1","Ala.","IV","3","South","6","East South Central","11"
"2","Alaska","AK","USA","state","10","current","occupied","","2","Alaska","X","4","West","9","Pacific","9"
"3","Arizona","AZ","USA","state","10","current","occupied","","4","Ariz.","IX","4","West","8","Mountain","9"
"4","Arkansas","AR","USA","state","10","current","occupied","","5","Ark.","VI","3","South","7","West South Central","8"
"5","California","CA","USA","state","10","current","occupied","","6","Calif.","IX","4","West","9","Pacific","9"
"6","Colorado","CO","USA","state","10","current","occupied","","8","Colo.","VIII","4","West","8","Mountain","10"
"7","Connecticut","CT","USA","state","10","current","occupied","","9","Conn.","I","1","Northeast","1","New England","2"
"8","Delaware","DE","USA","state","10","current","occupied","","10","Del.","III","3","South","5","South Atlantic","3"
"9","Florida","FL","USA","state","10","current","occupied","","12","Fla.","IV","3","South","5","South Atlantic","11"
"10","Georgia","GA","USA","state","10","current","occupied","","13","Ga.","IV","3","South","5","South Atlantic","11"
"""

PLACERESPONSE = """UA,UANAME,STATE,PLACE,PLNAME,CLASSFP,GEOID,POPPT,HUPT,AREAPT,AREALANDPT,UAPOP,UAHU,UAAREA,UAAREALAND,PLPOP,PLHU,PLAREA,PLAREALAND,UAPOPPCT,UAHUPCT,UAAREAPCT,UAAREALANDPCT,PLPOPPCT,PLHUPCT,PLAREAPCT,PLAREALANDPCT
00037,"Abbeville, LA Urban Cluster",22,00100,"Abbeville city",C1,2200100,12073,5168,13424306,13348680,19824,8460,29523368,29222871,12257,5257,15756922,15655575,60.9,61.09,45.47,45.68,98.5,98.31,85.2,85.26
00199,"Aberdeen--Bel Air South--Bel Air North, MD Urbanized Area",24,00125,"Aberdeen borough",C1,2400125,14894,6156,14961125,14942090,213751,83721,349451754,339626464,14959,6191,17618553,17599518,6.97,7.35,4.28,4.4,99.57,99.43,84.92,84.9
99999,"Not in a 2010 urban area",26,13480,"Carp Lake CDP",U1,2613480,357,526,12371409,5331938,,,,,357,526,12371409,5331938,,,,,100,100,100,100
00037,"Abbeville, LA Urban Cluster",22,99999,"Not in a census designated place or incorporated place",,2299999,3810,1537,10712370,10487499,19824,8460,29523368,29222871,,,,,19.22,18.17,36.28,35.89,,,,
62677,"New Orleans, LA Urbanized Area",22,01780,"Ama CDP",U1,2201780,1041,439,3021598,3016072,899703,426562,695715795,651105206,1316,547,11475232,9109388,.12,.1,.43,.46,79.1,80.26,26.33,33.11
01171,"Albuquerque, NM Urbanized Area",35,58070,"Placitas CDP (Sandoval County)",U1,3558070,544,280,2550047,2550047,741318,314851,657890843,648969769,4977,2556,76919539,76919539,.07,.09,.39,.39,10.93,10.95,3.32,3.32
77770,"St. Louis, MO--IL Urbanized Area",29,65000,"St. Louis city",C7,2965000,319293,176000,164556953,159893739,2150706,956440,2421404455,2392205874,319294,176002,171026250,160343174,14.85,18.4,6.8,6.68,100,100,96.22,99.72
43912,"Kansas City, MO--KS Urbanized Area",29,28090,"Grain Valley city",C1,2928090,12719,4818,13426555,13410717,1519417,671028,1773883282,1755587807,12854,4867,15720542,15704704,.84,.72,.76,.76,98.95,98.99,85.41,85.39
96670,"Winston-Salem, NC Urbanized Area",37,75000,"Winston-Salem city",C1,3775000,229432,103881,344210551,340988724,391024,174669,842062274,835485857,229617,103974,346269876,343041264,58.67,59.47,40.88,40.81,99.92,99.91,99.41,99.4
08785,"Boise City, ID Urbanized Area",16,08830,"Boise City city",C1,1608830,204776,92335,172985761,171285375,349684,146177,350800300,346614209,205671,92700,207328481,205550644,58.56,63.17,49.31,49.42,99.56,99.61,83.44,83.33
"""

@httpretty.activate
def test_negative_dict():
    httpretty.register_uri(
        httpretty.GET,
        PLACEURL,
        body=PLACERESPONSE,
        content_type='text/csv'
    )

    httpretty.register_uri(
        httpretty.GET,
        STATEURL,
        body=STATERESPONSE,
        content_type='text/csv'
    )

    results_states = negative_dict.__wrapped__()['states']
    assert results_states == {'alabama', 'al', 'alaska', 'ak', 'arizona', 'az', 'arkansas', 'ar', 'california', 'ca',
                              'colorado', 'co', 'connecticut', 'ct', 'delaware', 'de', 'florida', 'fl', 'georgia', 'ga'}

    results_places = negative_dict.__wrapped__()['places']
    assert results_places == {'abbeville', 'aberdeen', 'winston-salem', 'ama', 'placitas',
                              'boise city', 'grain valley', 'st. louis', 'carp lake'}
