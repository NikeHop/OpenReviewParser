"""
Utilities to interact with OpenReview Python API V2
"""

import logging
import time

import backoff
import getpass
import openreview

from urllib.error import HTTPError
from requests.exceptions import ConnectionError, Timeout

from openreview import OpenReviewException
from openreview.api import OpenReviewClient

from openreview_parser_v2.utils.data import VenueInstance


def get_openreview_client_v2(config: dict) -> OpenReviewClient:
    """
    Returns an instance of the OpenReviewClient class based on the provided configuration.

    Args:
        config (dict): A dictionary containing the configuration parameters.

    Returns:
        OpenReviewClient: An instance of the OpenReviewClient class.

    Raises:
        None

    """
    if config["guest_client"]:
        openreview_client = OpenReviewClient(baseurl="https://api2.openreview.net")
    else:
        username = input("OpenReview Username: ")
        password = getpass.getpass(prompt="OpenReview Password: ")
        openreview_client = OpenReviewClient(
            baseurl="https://api2.openreview.net", username=username, password=password
        )

    return openreview_client


def get_submissions(client: OpenReviewClient, venue: VenueInstance):
    """
    Retrieves the submissions for a given venue using the OpenReviewClient.

    Args:
        client (OpenReviewClient): An instance of the OpenReviewClient.
        venue (VenueInstance): An instance of the VenueInstance representing the venue.

    Returns:
        list: A list of submission notes for the given venue.

    Raises:
        OpenReviewException: If there is an error retrieving the submissions.
    """

    try:
        venue_group = client.get_group(venue.venue)

        if not hasattr(venue_group, "content"):
            logging.warning(
                f"The venue group {venue.venue} of does not have a content attribute."
            )
            return []

        submission_name = venue_group.content["submission_name"]["value"]
        submissions = client.get_all_notes(
            invitation=f"{venue.venue}/-/{submission_name}", details="directReplies"
        )

    except OpenReviewException as e:
        logging.error(f"Error getting submissions for {venue.venue}: {e}")
        return []

    return submissions


def on_backoff(details):
    error = details["exception"]
    info = error.args[0]
    message = info["message"]
    seconds_to_wait = int(message.split(" ")[-2])
    time.sleep(seconds_to_wait)


def giveup_code(details):
    return None


@backoff.on_exception(
    backoff.expo,
    (HTTPError, ConnectionError, Timeout, OpenReviewException),
    max_tries=5,
    on_backoff=on_backoff,
    on_giveup=giveup_code,
)
def get_notes(client: openreview.Client, **kwargs: str) -> list[openreview.Note]:
    """
    Wraps the get_all_notes function from the OpenReview client such that request limits are respected.

    Args:
        client (openreview.Client): OpenReview client
        logger (logging.Logger): Logger

    Returns:
        The list of OpenReview notes matching the **kwargs.
    """
    notes = client.get_notes(**kwargs)
    return notes
