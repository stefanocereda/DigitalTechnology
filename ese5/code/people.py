import requests

response = requests.get("http://api.open-notify.org/astros.json")

if response.status_code != 200:
    print("There was an error")
else:
    print("The answer is ", response.content)

import json
resp_dict = json.loads(response.content.decode('utf-8'))
# equivalent
resp_dict = response.json()

print("There are {} people in space".format(resp_dict['number']))
for astro in resp_dict['people']:
    print("{} is on the {}".format(astro['name'], astro['craft']))
