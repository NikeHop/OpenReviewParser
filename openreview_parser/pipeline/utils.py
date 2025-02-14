"""
Utilties 
"""

import logging

import backoff


from urllib.request import urlretrieve
from urllib.error import HTTPError, ContentTooShortError


def on_backoff(details):
    logging.info(
        f"Backing off {details['wait']:0.1f} seconds after {details['tries']} tries"
    )
    logging.info(f"Exception: {details['exception']}")


@backoff.on_exception(
    backoff.expo,
    (HTTPError, ContentTooShortError),
    max_tries=5,
    on_backoff=on_backoff,
    raise_on_giveup=False,
    on_giveup=lambda x: None,
)
def urlretrieve_backoff(url: str, filename: str) -> str | None:
    """
    Downloads a file from the given URL and saves it to the specified filename.

    Args:
        url (str): The URL of the file to download.
        filename (str): The name of the file to save.

    Returns:
        str | None: The path to the downloaded file if successful, None otherwise.
    """
    _, _ = urlretrieve(url, filename)
    return "success"
