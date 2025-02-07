"""
Utilities to interact with OpenReview Python API V1
"""

import getpass 

from  openreview import Client

def get_openreview_client_v1(config:dict)->Client:
    
    if config["guest_client"]:
            openreview_client = Client(baseurl="https://api.openreview.net")
    else:
        username = input("OpenReview Username: ")
        password = getpass.getpass(prompt="OpenReview Password: ")
        openreview_client = Client(
            baseurl="https://api.openreview.net", username=username, password=password
        )

    return openreview_client