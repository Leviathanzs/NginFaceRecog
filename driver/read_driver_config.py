import json


def readDriverConfig():
    # Opening JSON file
    fdriver = open('config/driverconf.json')

    # returns JSON object as
    # a dictionary
    drvconfdata = json.load(fdriver)

    # Closing file
    fdriver.close()

    return drvconfdata
