from skills_ml.algorithms.skill_extractors.noun_phrase_ending import \
    SkillEndingPatternExtractor, AbilityEndingPatternExtractor
from skills_ml.job_postings import JobPosting
from collections import namedtuple


posting_string = """Project Product Manager, ESPN.com Content
Job Description *This is a project position with an estimated length of 1 year*

The Product Manager will join the ESPN.com product management team. This role will directly manage key parts of the product vision, development and execution of the product experience for our Spanish language digital experience (e.g. Deportes). They will be responsible for creating a consistent and compelling experience across ESPN\u2019s Spanish language web, tablet and mobile experiences.

This role requires close collaboration with other vertical product owners, including ESPN.com web and native, platform, personalization, video, audio and more. The role requires strong spanish language skills and deductive reasoning ability. The role is based in Bristol, CT.

ESPN, Inc., The Worldwide Leader in Sports, is the leading multinational, multimedia sports entertainment company featuring the broadest portfolio of multimedia sports assets with over 50 business entities. Headquartered in Bristol, Connecticut, ESPN is 80% owned by ABC, Inc. (a subsidiary of The Walt Disney Company), and 20% by the Hearst Corporation.

ESPN was founded by Bill Rasmussen and launched on September 7, 1979. Now with over 6,500 employees, each year ESPN televises more than 5,100 live and/or original hours of sports programming. The company\u2019s mission is to serve sports fans. Anytime. Anywhere.
Responsibilities  
* Develop and maintain relationship with key members of ESPN\u2019s Spanish speaking editorial team 
* Responsible for prioritizing product features and executing key projects 
* Lead product process collaborating with design, engineering, editorial and business development organizations 
* Owner of product requirements documents (PRD), defining detailed specification of product goals, flows, experiences and designs 
* Stays on top of market trends to determine new or enhanced product capabilities that positively impact fan experience 
* Manage, develop and utilize a variety of tools for light proto-typing and conducting end user-testing and research 
* Manage day-to-day responsibility of product operation and coordination of product maintenance and support post product launch 
* Continually assesses and integrates internal and external customer feedback and business metrics
Basic Qualifications  
* A minimum of 2 years of work experience in product management 
* Demonstrated ability to work on a diverse scope of projects requiring detailed analysis, creative/practical problem solving, and sound judgment 
* Must have previous product ownership experience 
* Process driven and extremely detail oriented 
* Deep familiarity and passion around consumer web, application ecosystem, and mobile platforms 
* Must have strong analytical skills using Omniture, Google Analytics or related products 
* Strong communication skills 
* Be high energy and capable of multi-tasking 
* Ability to operate effectively in a team-oriented and collaborative environment 
* Fluency in speaking and writing in Spanish
Preferred Qualifications  
* Experience working in digital media and with editorial organizations a plus
Required Education  
* High School Diploma or equivalent work experience
Preferred Education  
* BA/BS degree preferred
Additional Information Imagine a career with an organization that brings smiles to millions every day. Imagine working with people whose passion for what they do is simply indescribable. We are The Walt Disney Company, live with a rich legacy of innovation, entertainment, and lifelong memories. With our vast array of both businesses and professionals, you\u2019ll have the opportunity to join a team that\u2019s beloved around the world, and to find out how it feels to love what you do. We invite you to discover for yourself why a career with Disney is the opportunity you\u2018ve been looking for.

ESPN is an equal opportunity employer \u2013 Female/Minority/Veteran/Disability. Our goal is to create an inclusive workplace for all.
Job Posting Industries Digital / Interactive
Job Posting Company ESPN
Primary Location-City Bristol
Primary Location-State CT
Primary Location-Country US
Auto req ID 274081BR"""
posting_object = {'id': '1234', '@type': 'JobPosting', 'description': posting_string}


def test_counts_skill_pattern_extractor():
    extractor = SkillEndingPatternExtractor()
    counts = extractor.document_skill_counts(posting_object)
    assert counts == {
        'strong analytical skills': 1,
        'strong communication skills': 1
    }

def test_candidates_skill_pattern_extractor():
    extractor = SkillEndingPatternExtractor()
    candidate_skills = sorted([cs for cs in extractor.candidate_skills(posting_object)], key=lambda x: x.matched_skill)
    assert candidate_skills[0].matched_skill == 'strong analytical skills'
    assert candidate_skills[0].context == '* Must have strong analytical skills using Omniture, Google Analytics or related products'
    assert candidate_skills[1].matched_skill == 'strong communication skills'
    assert candidate_skills[1].context == '* Strong communication skills'

def test_counts_skill_pattern_extractor_not_just_bullets():
    extractor = SkillEndingPatternExtractor(only_bulleted_lines=False)
    counts = extractor.document_skill_counts(posting_object)
    assert counts == {
        'strong analytical skills': 1,
        'strong communication skills': 1,
        'strong spanish language skills': 1
    }

def test_counts_ability_pattern_extractor_not_just_bullets():
    extractor = AbilityEndingPatternExtractor(only_bulleted_lines=False)
    counts = extractor.document_skill_counts(posting_object)
    assert counts == {
        'deductive reasoning ability': 1,
    }
