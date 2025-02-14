"""
Utilties to interact with the Semantic Scholar API.
"""

import logging
import os

from typing import TypedDict, Callable

import backoff
import requests

from openreview_parser.utils.data import Reference


def on_backoff_s2(details) -> None:
    logging.warning(f"Backing off {details['wait']}")
    print(details["exception"], type(details["exception"]))


def get_s2_header(use_api_key: bool) -> dict:
    """
    Retrieves the header for making API requests to Semantic Scholar.

    Returns:
        dict: The header containing the API key if it exists.
    """
    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", None)
    if use_api_key and api_key is not None:
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
    Returns the format of the id needed for semantic scholar for a given ID type.

    Parameters:
        id_type (str): The type of ID (e.g., "arxiv", "semantic_scholar").
        idd (str): The ID.

    Returns:
        str: The formatted ID.

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


@backoff.on_exception(
    backoff.expo,
    requests.exceptions.HTTPError,
    max_time=5,
    on_backoff=on_backoff_s2,
    raise_on_giveup=False,
)
def get_s2info(
    paper_title: str, paper_info: list[str], use_api_key: bool = False
) -> dict:
    """
    Retrieves information about a scientific paper from the Semantic Scholar API.

    Args:
        paper_title (str): The title of the paper.
        paper_info (list[str]): A list of information fields to retrieve for the paper.
        use_api_key (bool, optional): Whether to use an API key for accessing the Semantic Scholar API. Defaults to False.

    Returns:
        dict: A dictionary containing the retrieved information about the paper.
    """
    header = get_s2_header(use_api_key)

    response = requests.get(
        f"https://api.semanticscholar.org/graph/v1/paper/search/match?query={paper_title}",
        params={"fields": ",".join(paper_info)},
        headers=header,
    )
    response.raise_for_status()
    s2info = response.json()

    if "data" not in s2info:
        return {}
    else:
        if s2info["data"] is None:
            return {}
        else:
            return s2info["data"][0]


@backoff.on_exception(
    backoff.expo,
    requests.exceptions.RequestException,
    raise_on_giveup=False,
    max_time=5,
    on_backoff=on_backoff_s2,
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
    reference_info = response.json()

    # Parse response into Reference objects
    references = reference_info["data"]
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
