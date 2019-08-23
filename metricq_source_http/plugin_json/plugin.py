import json
import jsonpath_rw


def response_parse(response, **kwargs):
    json = json.load(response)
    if not 'json_path' in kwargs:
        raise Exception('missing required parameter "json_path"')
    jsonpath_expr = jsonpath_rw.parse(kwargs['json_path'])
    value_list = jsonpath_expr.find(json)
    if not len(value_list):
        raise Exception('no value found')
    return value_list[0].value

