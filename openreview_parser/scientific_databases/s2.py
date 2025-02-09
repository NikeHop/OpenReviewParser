import logging
import os

import backoff

from openreview_parser.utils.data import Reference


def get_citation_score(paper):
    paper


def get_s2_header(use_api_key: bool) -> dict:
    """
    Retrieves the header for making API requests to Semantic Scholar.

    Returns:
        str: The header containing the API key.
    """
    if use_api_key:
        api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
        return {"X-API-KEY": api_key}
    else:
        return {}


def giveup_on_404(error: HTTPError) -> bool:
    """
    Returns True if the HTTPError has status code 404.
    """
    return error.response.status_code == 404


@backoff.on_exception(
    backoff.expo, requests.exceptions.RequestException, giveup=giveup_on_404
)
def get_s2_references(
    idd: str, id_type: str, use_api_key: bool, max_n_references: int = 100
) -> list[Reference]:
    """
    Retrieves references for a given paper from the Semantic Scholar API.

    Args:
        idd (str): The identifier of the paper.
        id_type (str): The type of identifier used (e.g., "doi", "arxiv").
        max_n_references (int, optional): The maximum number of references to retrieve. Defaults to 100.

    Returns:
        list[Reference]: A list of Reference objects representing the retrieved references.
    """
    try:
        header = get_s2_header(use_api_key)
        formatted_id = get_formatted_id(id_type, idd)
        url = f"https://api.semanticscholar.org/graph/v1/paper/{formatted_id}/references?fields=title,abstract,intents,authors,isInfluential,isOpenAccess,openAccessPdf,externalIds&offset=0&limit={max_n_references}"
        data = make_s2_request(url, header)
    except Exception as e:
        logging.warning(f"Could not get references for {idd}: {e}")
        return []

    references = data["data"]
    parsed_references = []

    for reference in references:
        intents = reference["intents"]
        isInfluential = reference["isInfluential"]
        reference = reference["citedPaper"]

        if "externalIds" in reference and reference["externalIds"] is not None:
            external_ids = reference["externalIds"]
            corpus_id = external_ids.get("CorpusId", None)
            if corpus_id is not None:
                corpus_id = str(corpus_id)
            arxiv_id = external_ids.get("ArXiv", None)
        else:
            external_ids = None
            corpus_id = None
            arxiv_id = None

        title = reference["title"]
        abstract = reference.get("abstract", "")
        authors = [parse_s2_author(author) for author in reference["authors"]]
        paperhash = get_paperhash(title, authors)

        # Check whether necessary fields are present
        if title is None or paperhash is None or abstract is None or authors is None:
            logging.warning(f"Could not parse reference: {reference}")
            continue

        ref = Reference(
            paperhash=paperhash,
            title=title,
            abstract=abstract,
            authors=authors,
            intents=intents,
            isInfluential=isInfluential,
            s2_corpus_id=corpus_id,
            arxiv_id=arxiv_id,
            external_ids=external_ids,
        )
        parsed_references.append(ref)

    return parsed_references
