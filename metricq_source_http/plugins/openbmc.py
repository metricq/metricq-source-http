import json


def response_parse(response: str) -> float:
    json_data = json.loads(response)
    try:
        return float(json_data["data"]["Value"]) * pow(
            10, float(json_data["data"]["Scale"])
        )
    except KeyError:
        raise ValueError("missing [data][Value] and/or [data][Scale] in json response")
