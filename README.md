# OpenReview Parser


This repo contains a pipeline that parses all OpenReview submissions and reviews into a unified format and annotates them with metadata including:

* research_hypothesis (annotated via LLM)
* references (from [Semantic Scholar](https://www.semanticscholar.org/))
* citation counts for accepted papers (from [Semantic Scholar](https://www.semanticscholar.org/))

## Dependencies 


#### GROBID 

#### Environment API version 1

```
conda create -n openreview_parser_v1 python=3.11
conda activate openreview_parser_v1
pip install 
```

#### Environment API version 2

```
conda create -n openreview_parser_v2 python=3.11
conda
```
## Run pipeline 

There are two pipeline depending on the version of the OpenReview API used. That is because older submissions can only be retrieved using OpenReview API version 1 while more recent submissions can only be retrieved using API version 2.

The pipeline proceeds in the following steps:

* Retrieve all venues from OpenReview
* For each venue where submission and reviews are available build a schema of the venue and map the schema to the unified data model
* Get all submissions and reviews and parse them into the unified data model 
* Annotate the submissions with metadata:
    * Title and Abstract of References
    * Research Hypothesis 
    * Citation Counts


To run the pipeline activate the corresponding conda environment and run:

For pipeline API version 1:

```
python pipeline.py 
```

```
```


## Updates 

Timestamps at which the pipeline was run (datasets partially available [here](nhop/scientific-quality-score-prediction))

* Latest: 1.1.2025
