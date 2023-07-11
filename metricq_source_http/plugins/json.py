import json
from functools import lru_cache

import jsonpath_rw  # type: ignore

# The caching helps by
# 1. reducing the number of costly jsonpath_rw.parse calls
# 2. reducing the number of json.loads calls for the same response, which we cannot
# otherwise cache easily because of the plugin architecture.
# It does not seem to cost alot
# maxsize values are tuned to reasonable upper bounds for a single http source


@lru_cache(maxsize=256)
def _get_expr(json_path: str) -> jsonpath_rw.JSONPath:
    return jsonpath_rw.parse(json_path)


@lru_cache(maxsize=1024)
def _load_text(text: str) -> dict:
    return json.loads(text)


def response_parse(response: str, *, json_path: str) -> float:
    json_data = _load_text(response)
    jsonpath_expr = _get_expr(json_path)
    value_list = jsonpath_expr.find(json_data)
    if len(value_list) == 0:
        raise Exception("no value found")
    return float(value_list[0].value)
