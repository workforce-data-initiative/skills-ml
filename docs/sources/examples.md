# Usage Examples

## [Corpus Creator with Sampling and Filtering](https://github.com/workforce-data-initiative/skills-ml/blob/master/examples/Creating Corpus and Sampled Corpus.ipynb)

To showcase the corpus creator and its options, we generate a few different job postings corpora:

- a simple one from a single quarter's worth of data
- filtered on different fields like SOC code and base salary
- reservoir-sampled versions of each of the above

## [Extracting Skills using Noun Phrase Endings](https://github.com/workforce-data-initiative/skills-ml/blob/master/examples/NounPhraseSkillExtraction.py)

To showcase the noun phrase skill extractor, we download open job postings
from Virginia Tech's open data portal and run them through the skill extractor.
In the end, we have the most commonly occurring noun phrases ending in
'skill' or 'skills'.

## [Skill Extraction and Evaluation Loop](https://github.com/workforce-data-initiative/skills-ml/blob/master/examples/SkillExtractionEvaluation.py)

To showcase how skill extraction algorithms can be tested, we run extraction several times with different parameters:

- Skill extraction algorithms (exact, fuzzy matching)
- Base ontologies, consisting of ONET subsetted to Abilities, Skills, Knowledge)
- Metrics (Total Vocabulary Size, Total Candidate Skills, Recall of Given Ontology)


## [Train an Word2Vec Embedding Model using Quarterly Jobposting Data](https://github.com/workforce-data-initiative/skills-ml/blob/master/examples/TrainEmbedding.py)

To showcase the interface of training a word2vec embedding model in an online batch learning fashion:

- A list of quarters for creating the corpus from job posting data
- A trainer object that specifies some parameters of source, s3 path, batch size, model type ...etc.
- The train method takes whatever arugments `gensim.models.word2vec.Word2Vec` or `gensim.model.doc2vec.Doc2Vec` has

## [Compute and Aggregate Properties of Job Postings as a Tabular Dataset](https://github.com/workforce-data-initiative/skills-ml/blob/master/examples/ComputeAndAggregateJobPostingProperties.py)

To show job posting property computation and aggregation,
we calculate job posting counts by cleaned title, and upload
the resulting CSV to S3.

This is essentially a mini version of the Data@Work Research Hub.

To enable this example to be run with as few dependencies as possible, we use:

- a fake local s3 instance
- a sample of the Virginia Tech open job postings dataset
- only title cleaning and job counting.

## [Train an Occupation Classifier with Sample Jobposting Data](https://github.com/workforce-data-initiative/skills-ml/blob/master/examples/TrainOccupationClassifier.py)

To showcase how occupation classifier can be trained using grid search and cross-validation:

- A sample of the Virginia Tech open job postings dataset
- An embedding model that is trained already
- The pipeline objects that takes in all steps including filters, transformation, tokenization and vectorization
- A config dictionary for grid search
- A matrix object that specifies the data source, target variable, pipelines
- An occupation classifier trainer object that specifies input matrix, number of folds, grid search config, storage and number of workers
