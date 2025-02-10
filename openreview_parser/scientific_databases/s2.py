import logging
import os
import time

import backoff
import requests

from openreview_parser.utils.data import Reference


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


def clean_arxiv_id(arxiv_id: str) -> str:
    """
    Cleans the ArXiv ID by removing the version number if present.

    Args:
        arxiv_id (str): The ArXiv ID.

    Returns:
        str: The cleaned ArXiv ID.
    """
    if "v" in arxiv_id:
        arxiv_id = arxiv_id.split("v")[0]
    return arxiv_id


def get_formatted_id(id_type: str, idd: str) -> str:
    """
    Returns the URL for a given ID type and arXiv ID.

    Parameters:
        id_type (str): The type of ID (e.g., "arxiv", "semantic_scholar").
        idd (str): The ID.

    Returns:
        str: The URL corresponding to the given ID type and arXiv ID.

    Raises:
        NotImplementedError: If the provided ID type is not available.
    """

    if id_type == "arxiv":
        idd = clean_arxiv_id(idd)
        return f"ARXIV:{idd}"
    elif id_type == "semantic_scholar":
        return f"{idd}"
    elif id_type == "corpus_id":
        return f"CorpusId:{idd}"
    elif id_type == "doi":
        return f"doi:{idd}"
    else:
        raise NotImplementedError(f"The id_type {id_type} is not available")


def on_backoff(details):
    logging.warning(f"Backing off {details['wait']}")
    print(details["exception"], type(details["exception"]))


def giveup(details):
    return {}


@backoff.on_exception(
    backoff.expo,
    requests.exceptions.HTTPError,
    max_time=5,
    on_backoff=on_backoff,
    giveup=giveup,
    raise_on_giveup=False,
)
def get_s2info(paper_title, paper_info, use_api_key: bool = False) -> dict:
    if use_api_key:
        assert os.environ.get("S2_API_KEY") is not None, "S2 API key not found"
    header = get_s2_header(use_api_key)

    response = requests.get(
        f"https://api.semanticscholar.org/graph/v1/paper/search/match?query={paper_title}",
        params={"fields": ",".join(paper_info)},
        headers=header,
    )
    response.raise_for_status()
    response = response.json()

    if "data" not in response:
        return {}

    else:
        return response["data"]


def giveup_on_404(details):
    if details["exception"].response.status_code == 404:
        return True
    return False


@backoff.on_exception(
    backoff.expo,
    requests.exceptions.RequestException,
    raise_on_giveup=False,
    max_time=5,
    on_backoff=on_backoff,
    giveup=giveup,
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

    # Make s2 reuqest
    header = get_s2_header(use_api_key)
    formatted_id = get_formatted_id(id_type, idd)
    url = f"https://api.semanticscholar.org/graph/v1/paper/{formatted_id}/references?fields=title,abstract,intents,authors,isInfluential,isOpenAccess,openAccessPdf,externalIds&offset=0&limit={max_n_references}"
    response = requests.get(url, headers=header)
    response.raise_for_status()
    response = response.json()

    # Parse response into Reference objects
    references = response["data"]
    parsed_references = []

    for reference in references:
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

        title = reference.get("title", "")
        abstract = reference.get("abstract", "")

        if title == "" or abstract == "":
            continue

        intents = reference.get("intents", None)
        isInfluential = reference.get("isInfluential", None)
        reference = reference.get("citedPaper", None)
        authors = [author["name"] for author in reference["authors"]]
        external_ids = reference.get("externalIds", None)
        if external_ids is not None:
            corpus_id = external_ids.get("CorpusId", None)
            arxiv_id = external_ids.get("ArXiv", None)
        else:
            corpus_id = None
            arxiv_id = None

        ref = Reference(
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
