import json


def response_parse(response, **kwargs):
    json_data = json.loads(response)
    if "data" in json_data:
        if "Scale" in json_data["data"] and "Value" in json_data["data"]:
            value = json_data["data"]["Value"] * pow(10, json_data["data"]["Scale"])
        else:
            raise Exception('missing "Scale" and/or "Value" in json')
    else:
        raise Exception('missing "data" in json')
    return value
