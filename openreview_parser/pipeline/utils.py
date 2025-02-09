"""
Utilties 
"""

import logging

import backoff

from requests.exceptions import ConnectionError, Timeout

from urllib.request import urlretrieve
from urllib.error import HTTPError


def success_code(details):
    return "success"


def on_backoff(details):
    logging.info(
        f"Backing off {details['wait']:0.1f} seconds after {details['tries']} tries"
    )
    logging.info(f"Exception: {details['exception']}")


@backoff.on_exception(
    backoff.expo,
    (HTTPError, ConnectionError, Timeout),
    max_tries=5,
    on_backoff=on_backoff,
    on_success=success_code,
    raise_on_giveup=False,
    giveup=lambda x: None,
)
def urlretrieve_backoff(url: str, filename: str) -> str | None:
    return urlretrieve(url, filename)
