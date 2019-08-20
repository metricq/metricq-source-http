# metricq-source-http

A MetricQ source for HTTP servers

Currently supported formats: json, openbmc

Configuration is provided by the MetricQ management interface on startup. The configuration format (JSON) includes the following global keys:

``interval`` in seconds, the default request interval for all metrics. Can be overridden on a per-host or per-metric base.

``http_timeout`` in seconds, the timeout for HTTP requests. Should ideally be shorter than ``interval``.

``hosts`` object containing hostnames/IP adresses 

Host objects should define the following keys:

``name`` will be part (prefix) of the metric name

``login_type`` either "none", "basic", "cookie". More on login types below.

``insecure`` (optional) default: "false", falls back on HTTP instead of HTTPS if set to "true"

``metrics`` containng several metric keys with objects as values

Metric objects can contain the following keys:

``path`` HTTP path on the server, e.g ``/xyz/openbmc_project/senors/power/total_power``

``plugin`` either ``json`` or ``openbmc``. Additional plugins can be installed.

``plugin_params`` parameters for the plugin, e.g. ``json_path`` containing a JSONPath to the desired value

``unit``

``description``
