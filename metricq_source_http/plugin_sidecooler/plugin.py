import json
import re
from html.parser import HTMLParser

NaN = float("nan")


class SC_HTML(HTMLParser):
    kvs = {}

    def handle_data(self, data):
        if data.strip():
            sanitized = re.sub(r"\s", "", data)
            mo = re.match("(.+)=(.+);", sanitized)
            if mo:
                key = mo.group(1).strip()
                value = mo.group(2).strip()
                self.kvs[key] = value


def response_parse(response, **kwargs):
    key = kwargs["key"]
    data_type = kwargs["data_type"]
    parser = SC_HTML()
    parser.feed(response)
    if key in parser.kvs:
        raw_value = parser.kvs[key]
        if data_type.lower() == "number":
            return float(raw_value)
        elif data_type.lower() == "absolute":
            return abs(float(raw_value))
        elif data_type.lower() == "absolute_kilo":
            return abs(float(raw_value) * 1000)
        elif data_type.lower() == "state":
            return int(raw_value == "Yes")
    return NaN
