"""
Load venues 
"""

import json 
import logging 
import sys

import pandas as pd 
import tqdm 

from openreview_parser.scientific_databases.openreview_v2 import get_submissions, get_openreview_client_v2
from openreview_parser.utils.data import VenueInstance

def get_venue_v1():
    df = pd.read_csv("venues.csv")
    venues = list(df["venue"])

    with open("venues_v1.json","w") as f:
        json.dump(venues, f, indent=4)
    

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def check_which_venue_are_v1():
    with open("../dataset/venues/venues.json","r") as f:
        venues = json.load(f)
        venues = [VenueInstance(**vi) for vi in venues]
    
    client = get_openreview_client_v2({"guest_client":False})

    version1_venue_names = []

    for venue in tqdm.tqdm(venues):
        response = get_submissions(client,venue)
        if response == "version1":
            version1_venue_names.append(venue.venue)
    
    with open("venue_1_names.json","w") as f:
        json.dump(version1_venue_names, f, indent=4)


check_which_venue_are_v1()


