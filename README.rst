``metricq-source-http``
=======================

A MetricQ source for HTTP servers

Currently supported formats: json, openbmc

Configuration
------------

Configuration is provided by the MetricQ management interface on startup. The configuration format (JSON) includes the following global keys:

.. code-block:: json

   {
      "interval": <value>,
      "http_timeout": <value>,
      "hosts": { ... }   
   }

``interval`` in seconds, the default request interval for all metrics. Can be overridden on a per-host or per-metric base.

``http_timeout`` in seconds, the timeout for HTTP requests. Should ideally be shorter than ``interval``.

``hosts`` object containing hostnames/IP adress keys

Host objects should define the following keys:

.. code-block:: json

   "<address>": {
      "name": <metric-name-prefix>,
      "login_type": <value>,
      "metrics": { ... }  
   }

``name`` will be part (prefix) of the metric name

``login_type`` either ``"none"``, ``"basic"``, ``"cookie"``. More on login types below.

``metrics`` containng several metric keys with objects as values

``insecure`` (optional) default: ``"false"``, falls back on HTTP instead of HTTPS if set to ``"true"``

Metric objects should contain the following keys:


.. code-block:: json

   "<metric-name-suffix>": {
      "path": <host-name>,
      "plugin": <value>,
      "plugin_params": { ... }
   }

``path`` HTTP path on the server, e.g ``/xyz/openbmc_project/senors/power/total_power``

``plugin`` either ``json`` or ``openbmc``. Additional plugins can be installed.

``plugin_params`` parameters for the plugin, e.g. ``json_path`` containing a JSONPath to the desired value

``unit`` the unit of the metric, will be reported as metadata

``description`` a description of the resulting metric, metadata
