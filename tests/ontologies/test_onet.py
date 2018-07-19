from skills_ml.datasets.onet_cache import OnetSiteCache
from skills_ml.storage import InMemoryStore
from skills_ml.ontologies.onet import build_onet

onet_sample_lookup = {
    'Content Model Reference': """Element ID	Element Name	Description
1	Worker Characteristics	Worker Characteristics
1.A	Abilities	Enduring attributes of the individual that influence performance
1.A.1	Cognitive Abilities	Abilities that influence the acquisition and application of knowledge in problem solving
1.A.1.a	Verbal Abilities	Abilities that influence the acquisition and application of verbal information in problem solving
1.A.1.a.1	Oral Comprehension	The ability to listen to and understand information and ideas presented through spoken words and sentences.
1.A.1.a.2	Written Comprehension	The ability to read and understand information and ideas presented in writing.
1.A.1.a.3	Oral Expression	The ability to communicate information and ideas in speaking so others will understand.
2.A	Basic Skills	Developed capacities that facilitate learning or the more rapid acquisition of knowledge
2.A.1	Content	Background structures needed to work with and acquire more specific skills in a variety of different domains
2.A.1.a	Reading Comprehension	Understanding written sentences and paragraphs in work related documents.
2.A.1.b	Active Listening	Giving full attention to what other people are saying, taking time to understand the points being made, asking questions as appropriate, and not interrupting at inappropriate times.
2.A.1.c	Writing	Communicating effectively in writing as appropriate for the needs of the audience.
2.C	Knowledge	Organized sets of principles and facts applying in general domains
2.C.1	Business and Management	Knowledge of principles and facts related to business administration and accounting, human and material resource management in organizations, sales and marketing, economics, and office information and organizing systems
2.C.1.a	Administration and Management	Knowledge of business and management principles involved in strategic planning, resource allocation, human resources modeling, leadership technique, production methods, and coordination of people and resources.
2.C.1.b	Clerical	Knowledge of administrative and clerical procedures and systems such as word processing, managing files and records, stenography and transcription, designing forms, and other office procedures and terminology.
2.C.1.c	Economics and Accounting	Knowledge of economic and accounting principles and practices, the financial markets, banking and the analysis and reporting of financial data.""",
    'Knowledge': """O*NET-SOC Code	Element ID	Element Name	Scale ID	Data Value	N	Standard Error	Lower CI Bound	Upper CI Bound	Recommend Suppress	Not Relevant	Date	Domain Source
11-1011.00	2.C.1.a	Administration and Management	IM	4.75	27	0.09	4.56	4.94	N	n/a	07/2014	Incumbent
11-1011.00	2.C.1.a	Administration and Management	LV	6.23	27	0.17	5.88	6.57	N	N	07/2014	Incumbent
11-1011.00	2.C.1.b	Clerical	IM	2.66	27	0.22	2.21	3.11	N	n/a	07/2014	Incumbent
11-1011.00	2.C.1.b	Clerical	LV	3.50	27	0.41	2.66	4.34	N	N	07/2014	Incumbent
11-1011.00	2.C.1.c	Economics and Accounting	IM	3.70	27	0.28	3.11	4.28	N	n/a	07/2014	Incumbent""",
    'Skills': """O*NET-SOC Code	Element ID	Element Name	Scale ID	Data Value	N	Standard Error	Lower CI Bound	Upper CI Bound	Recommend Suppress	Not Relevant	Date	Domain Source
11-1011.00	2.A.1.a	Reading Comprehension	IM	4.12	8	0.13	3.88	4.37	N	n/a	07/2014	Analyst
11-1011.00	2.A.1.a	Reading Comprehension	LV	4.75	8	0.16	4.43	5.07	N	N	07/2014	Analyst
11-1011.00	2.A.1.b	Active Listening	IM	4.12	8	0.13	3.88	4.37	N	n/a	07/2014	Analyst
11-1011.00	2.A.1.b	Active Listening	LV	4.88	8	0.23	4.43	5.32	N	N	07/2014	Analyst
11-1011.00	2.A.1.c	Writing	IM	4.00	8	0.00	4.00	4.00	N	n/a	07/2014	Analyst
11-1011.00	2.A.1.c	Writing	LV	4.38	8	0.18	4.02	4.73	N	N	07/2014	Analyst""",
    'Abilities': """O*NET-SOC Code	Element ID	Element Name	Scale ID	Data Value	N	Standard Error	Lower CI Bound	Upper CI Bound	Recommend Suppress	Not Relevant	Date	Domain Source
11-1011.00	1.A.1.a.1	Oral Comprehension	IM	4.50	8	0.19	4.13	4.87	N	n/a	07/2014	Analyst
11-1011.00	1.A.1.a.1	Oral Comprehension	LV	4.88	8	0.13	4.63	5.12	N	N	07/2014	Analyst
11-1011.00	1.A.1.a.2	Written Comprehension	IM	4.25	8	0.16	3.93	4.57	N	n/a	07/2014	Analyst
11-1011.00	1.A.1.a.2	Written Comprehension	LV	4.62	8	0.18	4.27	4.98	N	N	07/2014	Analyst
11-1011.00	1.A.1.a.3	Oral Expression	IM	4.38	8	0.18	4.02	4.73	N	n/a	07/2014	Analyst
11-1011.00	1.A.1.a.3	Oral Expression	LV	5.00	8	0.00	5.00	5.00	N	N	07/2014	Analyst""",
    'Occupation Data': """O*NET-SOC Code	Title	Description
11-1011.00	Chief Executives	Determine and formulate policies and provide overall direction of companies or private and public sector organizations within guidelines set up by a board of directors or similar governing body. Plan, direct, or coordinate operational activities at the highest level of management with the help of subordinate executives and staff managers.
11-1011.03	Chief Sustainability Officers	Communicate and coordinate with management, shareholders, customers, and employees to address sustainability issues. Enact or oversee a corporate sustainability strategy.
11-1021.00	General and Operations Managers	Plan, direct, or coordinate the operations of public or private sector organizations. Duties and responsibilities include formulating policies, managing daily operations, and planning the use of materials and human resources, but are too diverse and general in nature to be classified in any one functional area of management or administration, such as personnel, purchasing, or administrative services.
11-1031.00	Legislators	Develop, introduce or enact laws and statutes at the local, tribal, State, or Federal level. Includes only workers in elected positions.
11-2011.00	Advertising and Promotions Managers	Plan, direct, or coordinate advertising policies and programs or produce collateral materials, such as posters, contests, coupons, or give-aways, to create extra interest in the purchase of a product or service for a department, an entire organization, or on an account basis.
11-2011.01	Green Marketers	Create and implement methods to market green products and services.""",
    'Tools and Technology': """O*NET-SOC Code	T2 Type	T2 Example	Commodity Code	Commodity Title	Hot Technology
11-1011.00	Tools	10-key calculators	44101809	Desktop calculator	N
11-1011.00	Tools	Desktop computers	43211507	Desktop computers	N"""
}

def test_build_onet():
    onet_sample_storage = InMemoryStore('')
    for filename, string in onet_sample_lookup.items():
        onet_sample_storage.write(string.encode('utf-8'), filename)

    onet_cache = OnetSiteCache(onet_sample_storage)
    ONET = build_onet(onet_cache)
    assert len(ONET.occupations) == 7

    # occupation list should have one major group and six detailed SOC
    assert len([occ for occ in ONET.occupations if len(occ.children) > 0]) == 1

    assert len([occ for occ in ONET.occupations if len(occ.parents) > 0]) == 6

    # should have 2 T2s, 2 commodities, 3 abilities, 3 skills, 3 knowledge
    assert len(ONET.competencies) == 2 + 2 + 3 + 3 + 3

    assert len([
        comp for comp in ONET.competencies    
        if 'O*NET T2' in comp.categories
    ]) == 2

    assert len([
        comp for comp in ONET.competencies    
        if 'O*NET T2' in comp.categories
        and 'UNSPSC Commodity' in list(comp.parents)[0].categories
    ]) == 2
