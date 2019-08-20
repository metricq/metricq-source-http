

def json_parse(json, **kwargs):
    if 'data' in json:
        if 'Scale' in json['data'] and 'Value' in json['data']:
            value = json['data']['Value'] * \
                pow(10, json['data']['Scale'])
        else:
            raise Exception('missing "Scale" and/or "Value" in json')
    else:
        raise Exception('missing "data" in json')
    return value