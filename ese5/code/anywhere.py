import requests
import datetime

pd = {'q': "Rome"}
#resp = requests.get("https://api.opencagedata.com/geocode/v1/json", params=pd)

#print(resp.status_code)
# Error 401 Unauthorized
# We need to register to the service and receive an api key

city = input("Where are you? ")
pd['q'] = city
# You need to register to opencagedata to receive a key
pd['key'] = "PUT YOUR KEY HERE"
resp = requests.get("https://api.opencagedata.com/geocode/v1/json", params=pd)
match = resp.json()['results'][0]

lat = match['geometry']['lat']
lon = match['geometry']['lng']
full_name = match['formatted']
flag = match['annotations']['flag']  # You need to have Unicode flags

pd = {'lat': lat, 'lon': lon}
resp = requests.get("http://api.open-notify.org/iss-pass.json", params=pd)
resp = resp.json()
print("The ISS will pass over {} {} at:".format(full_name, flag))
for iss_pass in resp['response']:
    dt = datetime.datetime.utcfromtimestamp(iss_pass['risetime'])
    print(dt.strftime('%Y-%m-%d %H:%M:%S'))
