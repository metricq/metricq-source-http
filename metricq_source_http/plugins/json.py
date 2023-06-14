import json

import jsonpath_rw  # type: ignore


def response_parse(response: str, *, json_path: str) -> float:
    json_data = json.loads(response)
    jsonpath_expr = jsonpath_rw.parse(json_path)
    value_list = jsonpath_expr.find(json_data)
    if len(value_list) == 0:
        raise Exception("no value found")
    return float(value_list[0].value)
