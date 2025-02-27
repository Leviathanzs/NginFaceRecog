import json


def readWorkerConfig():
    # Opening JSON file
    fworker = open('config/workerconfgrpc.json')

    # returns JSON object as
    # a dictionary
    wrkconfdata = json.load(fworker)
    # Closing file
    fworker.close()

    return wrkconfdata

def readWorkerConfigAsync():
    # Opening JSON file
    fworker = open('config/workerconf.json')

    # returns JSON object as
    # a dictionary
    wrkconfdata = json.load(fworker)
    # Closing file
    fworker.close()

    return wrkconfdata
