from skills_ml.algorithms.skill_extractors import SectionExtractSkillExtractor

def test_section_extract_skill_extractor_counts():
    sample_posting = {
        "description": "1st Assistant Golf Professional\n\n\n\nAll times are in Coordinated Universal Time.\n\nRequisition ID\n\n2017-14328\n\n# of Openings\n\n1\n\nJob Locations\n\nUS-TX-Austin\n\nPosted Date\n\n3/27/2017\n\nCategory (Portal Searching)\n\nGolf Resort Operations\n\n\n\n\n\nMore information about this job:\n\n\n\n\n\nLocation:\n\n\n\n\n\n\n\n**Barton Creek Resort & Spa**\n\n\n\n[Barton Creek]\n\n\n\nBarton Creek Resort & Spa's success is due to its dedicated, intelligent and self-motivated family of associates who work together to maintain the company's trademark high standards. If you would like to be a part of an environment where teamwork is emphasized and individual excellence is encouraged then this is the place for you.\n\n\n\nOmni Barton Creek Resort and Spa\u2019s associates enjoy a dynamic and exciting work environment, comprehensive training and mentoring, along with the pride that comes from working for a company with a reputation for exceptional service. We embody a culture of respect, gratitude and empowerment day in and day out. If you are a friendly, motivated person, with a passion to serve others, the Omni Barton Creek may be your perfect match.\n\n\n\n\n\nJob Description:\n\n\n\n\n\nThe 1st Assistant Golf Professional will assist the First Assistant and the Head Golf Professional with all activities relating to the management and execution of the properties golf operations, including but not limited to Member tournaments, Resort and outside events, outside service staff, golf shop, practice facilities, instructional program, financial management, human resource management and maintenance of golf equipment.\n\n\n\n****\n\n\n\n\n\nResponsibilities:\n\n\n\n\n\n\n\n* Ensure that all members and guests are greeted and welcomed in a professional and courteous manner.\n\n* Answer telephones to schedule future starting times and communicate information in a pleasant and professional manner.\n\n* Maintain inventory control and cash bank.\n\n* Ensure the golf course is properly marked.\n\n* Professionally communicate information, sell merchandise, and become fully knowledgeable in all products in the golf shop. Anticipate the needs of members and guests to offer appropriate merchandise alternatives.\n\n* Assist with physical inventories, as prescribed by Director of Retail and Head Golf Professional.\n\n* Ensure the \u201cpace of play\u201d standard is maintained daily.\n\n* Oversee rental club inventory.\n\n\n\n\n\n\n\n\n\n\n\nQualifications:\n\n\n\n\n\n\n\n* Excellent communication skills, both verbal and written\n\n* Solid computer skills including Microsoft Outlook, Word, and Excel\n\n* Proven leadership\n\n* Reputation for quality and attention to detail\n\n* Ability and willingness to work long hours and weekends as demanded by business cycles\n\n* Ability to use logical and rational thinking to solve problems.\n\n* Flexibility in hours, ability to manage multiple tasks",
        "jobLocation": {"address": {"addressRegion": "TX", "addressLocality": "Austin", "@type": "PostalAddress"}, "@type": "Place"},
        "@type": "JobPosting",
        "skills": "",
        "numPositions": "",
        "datePosted": 2011,
        "baseSalary": {"maxValue": "", "minValue": "", "salaryFrequency": "", "@type": "MonetaryAmount"},
        "onet_soc_code": "39-3091.00",
        "@context": "http://schema.org",
        "id": "ABC_12345",
        "industry": "",
        "experienceRequirements": "",
        "qualifications": "",
        "title": "1st Assistant Golf Professional",
        "educationRequirements": ""
    }

    skill_extractor = SectionExtractSkillExtractor('Qualifications')
    assert skill_extractor.document_skill_counts(sample_posting) == {
        'excellent communication skills both verbal and written': 1,
        'solid computer skills including microsoft outlook word and excel': 1,
        'proven leadership': 1,
        'reputation for quality and attention to detail': 1,
        'ability and willingness to work long hours and weekends as demanded by business cycles': 1,
        'ability to use logical and rational thinking to solve problems': 1,
        'flexibility in hours ability to manage multiple tasks': 1
    }


def test_section_extract_skill_extractor_candidate_skill():
    sample_posting = {
        "description": "1st Assistant Golf Professional\n\n\n\nAll times are in Coordinated Universal Time.\n\nRequisition ID\n\n2017-14328\n\n# of Openings\n\n1\n\nJob Locations\n\nUS-TX-Austin\n\nPosted Date\n\n3/27/2017\n\nCategory (Portal Searching)\n\nGolf Resort Operations\n\n\n\n\n\nMore information about this job:\n\n\n\n\n\nLocation:\n\n\n\n\n\n\n\n**Barton Creek Resort & Spa**\n\n\n\n[Barton Creek]\n\n\n\nBarton Creek Resort & Spa's success is due to its dedicated, intelligent and self-motivated family of associates who work together to maintain the company's trademark high standards. If you would like to be a part of an environment where teamwork is emphasized and individual excellence is encouraged then this is the place for you.\n\n\n\nOmni Barton Creek Resort and Spa\u2019s associates enjoy a dynamic and exciting work environment, comprehensive training and mentoring, along with the pride that comes from working for a company with a reputation for exceptional service. We embody a culture of respect, gratitude and empowerment day in and day out. If you are a friendly, motivated person, with a passion to serve others, the Omni Barton Creek may be your perfect match.\n\n\n\n\n\nJob Description:\n\n\n\n\n\nThe 1st Assistant Golf Professional will assist the First Assistant and the Head Golf Professional with all activities relating to the management and execution of the properties golf operations, including but not limited to Member tournaments, Resort and outside events, outside service staff, golf shop, practice facilities, instructional program, financial management, human resource management and maintenance of golf equipment.\n\n\n\n****\n\n\n\n\n\nResponsibilities:\n\n\n\n\n\n\n\n* Ensure that all members and guests are greeted and welcomed in a professional and courteous manner.\n\n* Answer telephones to schedule future starting times and communicate information in a pleasant and professional manner.\n\n* Maintain inventory control and cash bank.\n\n* Ensure the golf course is properly marked.\n\n* Professionally communicate information, sell merchandise, and become fully knowledgeable in all products in the golf shop. Anticipate the needs of members and guests to offer appropriate merchandise alternatives.\n\n* Assist with physical inventories, as prescribed by Director of Retail and Head Golf Professional.\n\n* Ensure the \u201cpace of play\u201d standard is maintained daily.\n\n* Oversee rental club inventory.\n\n\n\n\n\n\n\n\n\n\n\nQualifications:\n\n\n\n\n\n\n\n* Excellent communication skills, both verbal and written\n\n* Solid computer skills including Microsoft Outlook, Word, and Excel\n\n* Proven leadership\n\n* Reputation for quality and attention to detail\n\n* Ability and willingness to work long hours and weekends as demanded by business cycles\n\n* Ability to use logical and rational thinking to solve problems.\n\n* Flexibility in hours, ability to manage multiple tasks",
        "jobLocation": {"address": {"addressRegion": "TX", "addressLocality": "Austin", "@type": "PostalAddress"}, "@type": "Place"},
        "@type": "JobPosting",
        "skills": "",
        "numPositions": "",
        "datePosted": 2011,
        "baseSalary": {"maxValue": "", "minValue": "", "salaryFrequency": "", "@type": "MonetaryAmount"},
        "onet_soc_code": "39-3091.00",
        "@context": "http://schema.org",
        "id": "ABC_12345",
        "industry": "",
        "experienceRequirements": "",
        "qualifications": "",
        "title": "1st Assistant Golf Professional",
        "educationRequirements": ""
    }

    skill_extractor = SectionExtractSkillExtractor('Qualifications')
    candidate_skill = next(skill_extractor.candidate_skills(sample_posting))
    assert candidate_skill.skill_name == 'Excellent communication skills, both verbal and written'
    assert candidate_skill.start_index == 2370
