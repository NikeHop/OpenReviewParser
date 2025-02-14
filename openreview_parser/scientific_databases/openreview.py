"""Utilities to interact with OpenReview Python API V2."""

import logging
import time

import backoff
import getpass
import openreview

from urllib.error import HTTPError
from urllib3.exceptions import MaxRetryError, ResponseError
from requests.exceptions import ConnectionError, Timeout

from openreview import OpenReviewException, Client
from openreview.api import OpenReviewClient

from openreview_parser.utils.data import VenueInstance

ERRORS = (
    HTTPError,
    ConnectionError,
    Timeout,
    OpenReviewException,
    MaxRetryError,
    ResponseError,
)


import time


def on_backoff(details):
    """
    Handle backoff events in the OpenReview API.

    Args:
        details (dict): A dictionary containing details about the backoff event.

    Returns:
        None

    Raises:
        None
    """
    error = details["exception"]
    info = error.args[0]
    message = info["message"]
    seconds_to_wait = int(message.split(" ")[-2])
    time.sleep(seconds_to_wait)


def get_openreview_client(config: dict) -> OpenReviewClient | Client:
    """
    Return an instance of the OpenReviewClient class based on the provided configuration.

    Args:
        config (dict): A dictionary containing the configuration parameters.

    Returns:
        OpenReviewClient: An instance of the OpenReviewClient class.

    Raises:
        None
    """
    if config["use_openreview_api_v2"]:
        if config["openreview_guest_client"]:
            openreview_client = OpenReviewClient(baseurl="https://api2.openreview.net")
        else:
            username = input("OpenReview Username: ")
            password = getpass.getpass(prompt="OpenReview Password: ")
            openreview_client = OpenReviewClient(
                baseurl="https://api2.openreview.net",
                username=username,
                password=password,
            )
    else:
        if config["openreview_guest_client"]:
            openreview_client = Client(baseurl="https://api.openreview.net")
        else:
            username = input("OpenReview Username: ")
            password = getpass.getpass(prompt="OpenReview Password: ")
            openreview_client = Client(
                baseurl="https://api.openreview.net",
                username=username,
                password=password,
            )

    return openreview_client


def get_submissions(
    client: OpenReviewClient | Client, venue: VenueInstance, api_v2: bool = True
):
    """
    Retrieve the submissions for a given venue.

    Args:
        client (OpenReviewClient | Client): The OpenReview client or Client object.
        venue (VenueInstance): The venue for which to retrieve the submissions.
        api_v2 (bool, optional): Flag indicating whether to use the V2 API. Defaults to True.

    Returns:
        List[Submission]: The list of submissions for the given venue.
    """
    if api_v2:
        return get_submissions_v2(client, venue)
    else:
        return get_submissions_v1(client, venue)


@backoff.on_exception(
    backoff.expo,
    ERRORS,
    max_tries=5,
    on_backoff=on_backoff,
)
def get_submissions_v1(client: OpenReviewClient | Client, venue: VenueInstance):
    """
    Get the submissions for a given venue.

    Args:
        client (OpenReviewClient | Client): The OpenReview client used to retrieve the submissions.
        venue (VenueInstance): The venue for which to retrieve the submissions.

    Returns:
        list: A list of submission notes.
    """
    try:
        submissions = client.get_notes(
            invitation=f"{venue.venue}/-/Blind_Submission", details="directReplies"
        )
    except OpenReviewException as e:
        logging.error(f"Error getting submissions for {venue.venue}: {e}")
        return []

    return submissions


@backoff.on_exception(
    backoff.expo,
    ERRORS,
    max_tries=5,
    on_backoff=on_backoff,
)
def get_submissions_v2(client: OpenReviewClient | Client, venue: VenueInstance):
    """
    Retrieve the submissions for a given venue using the OpenReviewClient.

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


@backoff.on_exception(
    backoff.expo,
    ERRORS,
    max_tries=5,
    on_backoff=on_backoff,
)
def get_notes(client: openreview.Client, **kwargs: str) -> list[openreview.Note]:
    """
    Wrap the get_all_notes function from the OpenReview client such that request limits are respected.

    Args:
        client (openreview.Client): OpenReview client
        logger (logging.Logger): Logger

    Returns:
        The list of OpenReview notes matching the **kwargs.
    """
    notes = client.get_notes(**kwargs)
    return notes
