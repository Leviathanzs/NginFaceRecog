import json

# Opening JSON file
fdriver = open('config/driverconf.json')

# returns JSON object as
# a dictionary
drconfdata = json.load(fdriver)

# Iterating through the json
# list
for worker in drconfdata['workers']:
    print(worker['workername'])

# Closing file
fdriver.close()

# for each worker
#   connect to each worker
#   get list of executors in each worker
#   close connection
#
# executor_list

# read from database
# while loop
#   fetch rows into numpy array until 128 MB
#   send numpy array to an executor in executor_list
#
