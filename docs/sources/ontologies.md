# Working With Ontologies

skills-ml is introducing the CompetencyOntology class, for a rich, flexible representation of competencies, occupations, and their relationships with each other. The CompetencyOntology class is backed by JSON-LD, and based on Credential Engine's [CTDL-ASN format for Competencies](https://credreg.net/ctdlasn/terms#Competency). The goal is to be able to read in any CTDL-ASN framework and produce a CompetencyOntology object for use throughout the skills-ml library.

Furthermore, skills-ml contains pre-mapped versions of open frameworks like ONET for use out of the box.

## Competency

A competency, in the CTDL-ASN context, refers some knowledge, skill, or ability that a person can possess or learn. Each competency contains:

- A unique identifier within the ontology. If you're familiar with ONET, think the table of contents identifiers (e.g. '1.A.1.a.1')
- Some basic textual information: a name (e.g. Oral Comprehension) and/or description (e.g. 'The ability to listen to and understand information and ideas presented through spoken words and sentences.'),
, and maybe a general textual category (e.g. Ability)
- Associative information with other competencies. A basic example is a parent/child relationship, for instance ONET's definition of 'Oral Comprehension' as the child of another competency called 'Verbal Abilities'. CTDL-ASN encodes this using the 'hasChild' and 'isChildOf' properties, and this is used in skills-ml. There many other types of associations competencies can have with each other that the Competency class in skills-ml does not yet address, you can read more at the [Credential Engine's definition ofCompetency](http://purl.org/ctdlasn/terms/Competency).

The Competency class tracks all of this. It can be created using either keyword arguments in the class' Constructor or through a class method that loads from JSON-LD. 

### Basic Example

*Using Python Constructor*
```python
from skills_ml.ontologies import Competency

dinosaur_riding = Competency(
	identifier='12345',
	name='Dinosaur Riding',
	description='Using the back of a dinosaur for transportation'
)
```

*Using JSON-LD*
```python
from skills_ml.ontologies import Competency

dinosaur_riding = Competency.from_jsonld({
	'@type': 'Competency',
	'@id': '12345',
	'name': 'Dinosaur Riding',
	'description': 'Using the back of a dinosaur for transportation'
})
```
To aid in bi-directional searching, the Competency object is meant to include a parent/child relationshiop on both the parent and child objects. The add_parent and add_child methods modify both the parent and child objects to easily maintain this bi-directional relationship.

### Example parent/child relationship

*Using Python Constructor*

```python
from skills_ml.ontologies import Competency

dinosaur_riding = Competency(
	identifier='12345',
	name='Dinosaur Riding',
	description='Using the back of a dinosaur for transportation'
)

extreme_transportation = Competency(
	identifier='123',
	name='Extreme Transportation',
	description='Comically dangerous forms of transportation'
)
dinosaur_riding.add_parent(extreme_transportation)
print(dinosaur_riding.parents)
print(extreme_transportation.children)
```

*Using JSON-LD*

```python
dinosaur_riding = Competency.from_jsonld({
	'@type': 'Competency',
	'@id': '12345',
	'name': 'Dinosaur Riding',
	'description': 'Using the back of a dinosaur for transportation',
	'isChildOf': [{'@type': 'Competency', '@id': '123'}]
})

extreme_transportation = Competency.from_jsonld({
	'@type': 'Competency',
	'@id': '123',
	'name': 'Extreme Transportation',
	'description': 'Comically dangerous forms of transportation',
	'hasChild': [{'@type': 'Competency', '@id': '12345'}]
```

## Occupation
An Occupation is a job or profession that a person can hold. CTDL-ASN does not define this, so skills-ml models the Occupation similarly to the Competency, albeit with far less detail. 

- A unique identifier within the ontology. If you're familiar with ONET, think of an ONET SOC code (11-1011.00)
- Some basic textual information: a name (e.g. Civil Engineer), maybe a description.
- Associative information with other occupations. So far the only relationship modeled in skills-ml between occupations is a parent/child one, similarly to Competency. Going back to the ONET example, an occupation representing the major group (identifier 11) may be thought of as the parent of SOC code 11-1011.00.

### Basic Example

*Using Python Constructor*
```python
from skills_ml.ontologies import Occupation

dinosaur_rider = Occupation(
	identifier='9999',
	name='Dinosaur Rider',
)
```

*Using JSON-LD*
```python
from skills_ml.ontologies import Occupation

dinosaur_rider = Occupation.from_jsonld({
	'@type': 'Occupation',
	'@id': '9999',
	'name': 'Dinosaur Rider'
})
```

## CompetencyOccupationEdge

A CompetencyOccupationEdge is simply a relationship between a Competency and an Occupation. Currently, tthere are no further properties defined on this edge, though this will likely change in the future. 

### Basic Example

*Using Python Constructor*
```python
from skills_ml.ontologies import CompetencyOccupationEdge

CompetencyOccupationEdge(
	occupation=dinosaur_rider,
	competency=dinosaur_riding
)
```

*Using JSON-LD*
```python
from skills_ml.ontologies import CompetencyOccupationEdge

CompetencyOccupationEdge.from_jsonld({
	'@type': 'CompetencyOccupationEdge',
	'@id': 'competency=12345;occupation=9999',
	'competency': {'@type': 'Competency', '@id': '12345'},
	'occupation': {'@type': 'Occupation', '@id': '9999'}
})
```

## CompetencyFramework

A `CompetencyFramework` represent a collection of competencies and some metadata about them. The identifiers for given Competencies are used to disambiguate between them. The metadata exists so any code that uses the `CompetencyFramework` object can pass on useful knowledge about the framework to its output.

The metadata has only two pieces of data:
- name: A machine-readable name. Should be in snake case (e.g. `onet_ksat`)
- description: A human-readable description.

### Basic Example

*Using Python Constructor*

```python
from skills_ml.ontologies import Competency, CompetencyFramework

framework = CompetencyFramework(
    name='Sample Framework',
    description='A few basic competencies',
    competencies=[
        Competency(identifier='a', name='Organization'),
        Competency(identifier='b', name='Communication Skills'),
        Competency(identifier='c', name='Cooking')
    ]
)
```

## CompetencyOntology

An ontology represents a collection of competencies, a collection of occupations, and a collection of all relationships between competencies and occupations. The CompetencyOntology class represents each of these three collections using a `set` object. The identifiers for all of those objects are used to disambiguate between items in each of these sets. The JSON-LD representation of the ontology mirrors this internal structure.

Below is an example of the objects defined above arranged into a CompetencyOntology. For brevity, the descriptions are omitted. 

Note in the Python example that importing the CompetencyOccupationEdge class is not necessary when using the Ontology; the `add_edge` method of Ontology can simply take a competency and occupation directly. 

### Basic Example

*Using Python Constructor*

```python
from skills_ml.ontologies import Competency, Occupation, CompetencyOntology

ontology = CompetencyOntology(
    competency_name='caveman_games',
    competency_description='Competencies Useful to Characters in NES title Caveman Games'
)

dinosaur_riding = Competency(identifier='12345', name='Dinosaur Riding')
extreme_transportation = Competency(identifier='123', name='Extreme Transportation')
dinosaur_riding.add_parent(extreme_transportation)


dinosaur_rider = Occupation(identifier='9999', name='Dinosaur Rider')

ontology.add_competency(dinosaur_riding)
ontology.add_competency(extreme_transportation)
ontology.add_occupation(dinosaur_rider)
ontology.add_edge(occupation=dinosaur_rider, competency=dinosaur_riding)
```

*Using JSON-LD*

```python
from skills_ml.ontologies import CompetencyOntology

ontology = CompetencyOntology.from_jsonld({
	'competencies': [{
		'@type': 'Competency',
		'@id': '12345',
		'name': 'Dinosaur Riding',
		'description': 'Using the back of a dinosaur for transportation',
		'isChildOf': [{'@type': 'Competency', '@id': '123'}]
	}, {
		'@type': 'Competency',
		'@id': '123',
		'name': 'Extreme Transportation',
		'description': 'Comically dangerous forms of transportation',
		'hasChild': [{'@type': 'Competency', '@id': '12345'}]
	}],
	'occupations': [{
		'@type': 'Occupation',
		'@id': '9999',
		'name': 'Dinosaur Rider'
	}],
	'edges': [{
		'@type': 'CompetencyOccupationEdge',
		'@id': 'competency=12345;occupation=9999',
		'competency': {'@type': 'Competency', '@id': '12345'},
		'occupation': {'@type': 'Occupation', '@id': '9999'}
	}]
})
```

## Included Ontologies

### ONET

The `skills_ml.ontologies.onet` module contains a Onet class inherited from CompetencyOntology which will build the ontology during the instantiation from a variety of files on the ONET site, using at the time of writing the latest version of onet (db_v22_3):

- Content Model Reference.txt
- Knowledge.txt
- Skills.txt
- Abilities.txt
- Tools and Technology.txt
- Occupation Data.txt

```python

from skills.ml.ontologies.onet import Onet

ONET = Onet()
# this will take a while as it downloads the relatively large files and processes them
ONET.filter_by(lambda edge: 'forklift' in edge.competency.name)
```

If you pass in an ONET cache object, the raw ONET files can be cached on your filesystem so that building it the second time will be faster.

```python
from skills_ml.storage import FSStore
from skills_ml.datasets.onet_cache import OnetSiteCache
from skills_ml.ontologies.onet import Onet

ONET = Onet(OnetSiteCache(FSStore('onet_cache')))
```


## Uses of Ontologies

### Filtering

You can filter the graph to produce subsets based on the list of edges. This will return another CompetencyOntology object, so any code that takes an ontology as input will work on the subsetted graph.

You can optionally supply `competency_name` and `competency_description` keyword arguments to apply to the `CompetencyFramework` in the returned ontology object. This is necessary if you wish to send the `CompetencyFramework` object in the resulting ontology to algorithms in skills-ml.


```python

# Return an ontology that consists only of competencies with 'python' in the name, along with their related occupations
ontology.filter_by(lambda edge: 'python' in edge.competency.name.lower())

# Return an ontology that consists only of occupations with 'software' in the name, along with their associated competencies
ontology.filter_by(lambda edge: 'software' in edge.competency.name.lower())

# Return an ontology that is the intersection of 'python' competencies and 'software' occupations
ontology.filter_by(lambda edge: 'software' in edge.occupation.name.lower() and 'python' in edge.competency.name.lower())

# Return only competencies who have a parent competency containing 'software'
ontology.filter_by(lambda edge: any('software' in parent.name.lower() for parent in edge.parents)

# Return an ontology with only 'python' competencies, and set a name/description for the resulting CompetencyFramework
ontology.filter_by(lambda edge: 'python' in edge.competency.name.lower(), competency_name='python', competency_description='Python-related competencies')
```

### Skill Extraction: competencies-only

Many list-based skill extraction require a CompetencyFramework as input. This can be retrieved directory from the `CompetencyOntology` object. 

```python
from skills_ml.algorithms.skill_extractors import ExactMatchSkillExtractor
from skills_ml.job_postings.common_schema import JobPostingCollectionSample
skill_extractor = ExactMatchSkillExtractor(ontology.competency_framework)
for candidate_skill in skill_extractor.candidate_skills(JobPostingCollectionSample()):
    print(candidate_skill)
```

### Skill Extraction: filtered competencies

If you wish to filter a `CompetencyOntology` and then use it for skill extraction, you must make sure it has a name and description, either through the optional `filter_by` keyword argument or through modifying the `CompetencyFramework` instance directly.

```python

from skills_ml.algorithms.skill_extractors import ExactMatchSkillExtractor
from skills_ml.job_postings.common_schema import JobPostingCollectionSample


# Option 1: Using filter_by keyword arguments (recommended)

competency_framework = ontology.filter_by(
    lambda edge: 'python' in edge.competency.name.lower(),
    competency_name='python',
    competency_description='Python-related competencies'
).competency_framework


# Option 2: Modifying competency_framework afterwards
competency_framework = ontology.filter_by(lambda edge: 'python' in edge.competency.name.lower()).competency_framework
competency_framework.name = 'python'
competency_framework.description = 'Python-related competencies'


skill_extractor = ExactMatchSkillExtractor(competency_framework)

for candidate_skill in skill_extractor.candidate_skills(JobPostingCollectionSample()):
    print(candidate_skill)

```

### Skill Extraction: full ontology

The SocScopedExactMatchSkillExtractor requires both occupation and competency data, so it takes in the entire `CompetencyOntology` as input.

```python
from skills_ml.algorithms.skill_extractors import ExactMatchSkillExtractor
from skills_ml.job_postings.common_schema import JobPostingCollectionSample
skill_extractor = SocScopedExactMatchSkillExtractor(ontology)
for candidate_skill in skill_extractor.candidate_skills(JobPostingCollectionSample()):
    print(candidate_skill)
```

### Exporting as JSON-LD
You can export an ontology as a JSON-LD object for storage that you can later import

```python

import json

with open('out.json', 'w') as f:
	json.dump(ontology.jsonld, f)
```
