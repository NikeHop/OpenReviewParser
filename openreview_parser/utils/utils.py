"""
Utilties 
"""

import backoff 


backoff.on_exception(
    backoff.expo,
    (urllib.error.URLError, ConnectionError),
    max_tries=5,
    giveup=fatal_code
)
def urlretrieve_backoff()