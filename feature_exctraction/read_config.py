import json
import os

def readConfig():
    # Opening JSON file
    fworker = open(os.path.dirname(os.path.abspath(__file__))+'/config/featureextract_conf.json')

    # returns JSON object as
    # a dictionary
    data = json.load(fworker)
    # Closing file
    fworker.close()

    return data