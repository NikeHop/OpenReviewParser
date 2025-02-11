# Data Models

The data models can be found in `data.py` as [Pydantic](https://docs.pydantic.dev/latest/) BaseModels.

## Paper

| Attribute                | Type                                | Explanation                                                                 |
|--------------------------|-------------------------------------|-----------------------------------------------------------------------------|
| **Basic Paper Info**     |                                     |                                                                             |
| `title`                  | `str`                               | The title of the paper.                                                     |
| `authors`                | `list[str]`                         | A list of authors of the paper.                                             |
| `abstract`               | `str \| None`                       | The abstract of the paper (optional).                                       |
| `summary`                | `str \| None`                       | A summary of the paper (optional).                                          |
| **ID's**                 |                                     |                                                                             |
| `paperhash`              | `str`                               | A unique hash identifier for the paper. (last_name_first_author\|title)                                    |
| `arxiv_id`               | `str \| None`                       | The arXiv ID of the paper (optional).                                       |
| `s2_corpus_id`           | `str \| None`                       | The Semantic Scholar (S2) corpus ID of the paper (optional).                |
| **OpenReview Metadata**  |                                     |                                                                             |
| `field_of_study`         | `list[str] \| str \| None`          | The field(s) of study the paper belongs to (optional).                      |
| `venue`                  | `str \| None`                       | The venue where the paper was published (optional).                         |
| `publication_date`       | `str \| None`                       | The publication date of the paper (optional).                              |
| **Semantic Scholar Metadata**          |                                     |                                                                             |
| `n_references`           | `int \| None`                       | The number of references in the paper (optional).                          |
| `n_citations`            | `int \| None`                       | The number of citations the paper has received (optional).                 |
| `n_influential_citations`| `int \| None`                       | The number of influential citations the paper has received (optional).      |
| `external_ids`           | `dict \| None`                      | External IDs associated with the paper (optional).                         |
| **Content**              |                                     |                                                                             |
| `parsed_pdf`             | `dict \| None`                      | Parsed PDF content of the paper (optional).                                |
| `structured_content`     | `dict[str, Section]`                | Structured content of the paper, organized by sections.                    |
| **Review Data**          |                                     |                                                                             |
| `decision`               | `bool \| None`                      | The decision on the paper (e.g., accepted/rejected) (optional).            |
| `decision_text`          | `str \| None`                       | The text explaining the decision (optional).                               |
| `reviews`                | `list[Review] \| None`              | A list of reviews for the paper (optional).                                |
| `comments`               | `list[Comment] \| None`             | A list of comments on the paper (optional).                                |
| **References**           |                                     |                                                                             |
| `references`             | `list[Reference] \| None`           | A list of references cited in the paper (optional).                        |
| `section_name2section`   | `dict`                              | A mapping of section names to their corresponding sections.                |
| `bibref2section`         | `dict`                              | A mapping of bibliography references to their corresponding sections.      |
| `bibref2paperhash`       | `dict`                              | A mapping of bibliography references to their corresponding paper hashes.  |
| **Hypothesis**           |                                     |                                                                             |
| `hypothesis`             | `str \| None`                       | The hypothesis proposed in the paper annotated via an LLM (optional).                           |


## Review 

| Attribute                | Type                                | Explanation                                                                 |
|--------------------------|-------------------------------------|-----------------------------------------------------------------------------|
| `review_id`              | `str`                               | A unique identifier for the review.                                         |
| `review`                 | `TextReview`                        | The content of the review, represented by the `TextReview` model.           |
| `score`                  | `float \| None`                     | The overall score given by the reviewer (optional).                         |
| `confidence`             | `float \| None`                     | The reviewer's confidence in their assessment (optional).                   |
| `novelty`                | `float \| None`                     | The novelty score of the paper (optional).                                  |
| `correctness`            | `float \| None`                     | The correctness score of the paper (optional).                              |
| `clarity`                | `float \| None`                     | The clarity score of the paper (optional).                                  |
| `impact`                 | `float \| None`                     | The impact score of the paper (optional).                                   |
| `reproducibility`        | `float \| None`                     | The reproducibility score of the paper (optional).                          |
| `ethics`                 | `str \| None`                       | Ethical considerations noted by the reviewer (optional).                    |

## TextReview

| Attribute                | Type                                | Explanation                                                                 |
|--------------------------|-------------------------------------|-----------------------------------------------------------------------------|
| `title`                  | `str \| None`                       | The title of the review (optional).                                         |
| `paper_summary`          | `str \| None`                       | A summary of the paper being reviewed (optional).                           |
| `main_review`            | `str \| None`                       | The main content of the review (optional).                                  |
| `strength_weakness`      | `str \| None`                       | A section discussing the strengths and weaknesses of the paper (optional).  |
| `questions`              | `str \| None`                       | Questions raised by the reviewer (optional).                                |
| `limitations`            | `str \| None`                       | Limitations of the paper as noted by the reviewer (optional).               |
| `review_summary`         | `str \| None`                       | A summary of the review (optional).                                         |

## Comment

| Attribute                | Type                                | Explanation                                                                 |
|--------------------------|-------------------------------------|-----------------------------------------------------------------------------|
| `title`                  | `str \| None`                       | The title of the comment (optional).                                        |
| `comment`                | `str`                               | The content of the comment.                                                 |

## Reference

| Attribute                | Type                                | Explanation                                                                 |
|--------------------------|-------------------------------------|-----------------------------------------------------------------------------|
| **Basic Paper Info**     |                                     |                                                                             |
| `title`                  | `str`                               | The title of the referenced paper.                                          |
| `abstract`               | `str`                               | The abstract of the referenced paper (default is an empty string).          |
| `authors`                | `list[str]`                         | A list of authors of the referenced paper.                                  |
| **IDs**                  |                                     |                                                                             |
| `arxiv_id`               | `str \| None`                       | The arXiv ID of the referenced paper (optional, default is an empty string).|
| `s2_corpus_id`           | `str \| None`                       | The Semantic Scholar (S2) corpus ID of the referenced paper (optional, default is an empty string). |
| `external_ids`           | `dict \| None`                      | External IDs associated with the referenced paper (optional, default is an empty dictionary). |
| **Reference Specific Info** |                                  |                                                                             |
| `intents`                | `list[str] \| None`                 | The intents or purposes of the reference (optional).                        |
| `isInfluential`          | `bool \| None`                      | Indicates whether the reference is influential (optional).                  |




