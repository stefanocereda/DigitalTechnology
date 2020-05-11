import requests

params_dict = {
    'lat': 45.478156,
    'lon': 9.227080
}

resp = requests.get("http://api.open-notify.org/iss-pass.json", params=params_dict)
resp = resp.json()
print("The ISS will pass over PoliMi at {}".format(resp['response'][0]['risetime']))

import datetime
for iss_pass in resp['response']:
    dt = datetime.datetime.utcfromtimestamp(iss_pass['risetime'])
    print("ISS will pass over PoliMi at {} UTC".format(dt.strftime('%Y-%m-%d %H:%M:%S')))
