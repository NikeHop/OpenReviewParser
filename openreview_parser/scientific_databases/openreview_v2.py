"""
Utilities to interact with OpenReview Python API V2
"""

import logging 

import getpass

from openreview import OpenReviewException
from openreview.api import OpenReviewClient

from openreview_parser.utils.data import VenueInstance

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

def get_submissions(client: OpenReviewClient , venue:VenueInstance):
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
            logging(f"Does not have a content attribute {venue.venue}")
            return []

        submission_name = venue_group.content["submission_name"]['value']
        submissions = client.get_all_notes(invitation=f"{venue.venue}/-/{submission_name}",details="directReplies")
    
    except OpenReviewException as e:
        logging.error(f"Error getting submissions for {venue.venue}: {e}")
        return []

    return submissions
