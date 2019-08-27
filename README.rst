``metricq-source-http``
=======================

A MetricQ source for HTTP servers

Configuration
------------

Configuration is provided by the MetricQ management interface on startup. The configuration format (JSON) includes the following global keys:

.. code-block:: json

   {
      "interval": <value>,
      "http_timeout": <value>,
      "hosts": [ ... ]   
   }

``interval`` in seconds, the default request interval for all metrics. Can be overridden on the metric level.

``http_timeout`` in seconds, the timeout for HTTP requests. Should ideally be shorter than ``interval``.

``hosts`` list containing host objects.

Host objects should define the following keys:

.. code-block:: json

   {
      "hosts": <addresses>,
      "names": <metric-name-prefix's>,
      "login_type": <value>,
      "user": <value>,
      "password": <value>,
      "metrics": { ... }  
   }

``hosts`` is a hostrange string or list of hosts must be the same length as names

``names`` is a hostrange string or list of names, will be part (prefix) of the metric name

``login_type`` either ``"none"``, ``"basic"``, ``"cookie"``. More on login types below.

``user`` (optional) username for login_types ``"basic"`` and ``"cookie"``

``password`` (optional) password for login_types ``"basic"`` and ``"cookie"``

``metrics`` containing several metric keys with objects as values.

``insecure`` (optional) default: ``"false"``, falls back on HTTP instead of HTTPS if set to ``"true"``.

Metric objects should contain the following keys:

.. code-block:: json

   "<metric-name-suffix>": {
      "path": <host-name>,
      "plugin": <value>,
      "plugin_params": { ... }
   }

``path`` HTTP path on the server, e.g ``"/xyz/openbmc_project/senors/power/total_power"``

``plugin`` either ``"json"`` or ``"openbmc"``. Additional plugins can be installed.

``plugin_params`` parameters for the plugin, e.g. ``json_path`` containing a JSONPath to the desired value.

``unit`` the unit of the metric, will be reported as metadata.

``description`` a description of the resulting metric, metadata.

``interval`` in seconds.  Can be set to override the global default interval.

Login Types
~~~~~~~~~~~

``"none"`` no authentication necessary.

``"basic"`` HTTP Basic authentication headers (rfc7617), will be transmitted on every request. Requires the host keys ``user`` and ``password`` to also be specified.

``"cookie"`` a seperate login endpoint will be called via a POST request and a cookie be saved that contains a login session. Must be configured via additional parameters: ``login_path``. Requires the host keys ``user`` and ``password`` to also be specified.

Plugins
~~~~~~~

The source comes with two pre-installed plugins: 

* ``"json"`` for generic JSON data. Available ``plugin_params``:

   - ``json_path`` a JSONPath pointing to the value to be reported
     
* ``"openbmc"`` for the OpenBMC interface

Example
~~~~~~~

Querying power and temperature from a Redfish-capable server:

.. code-block:: json

  "interval": 60,
  "http_timeout": 15,
  "hosts": {
    "192.168.0.100": {
      "name": "MyRedfishServer",    
      "login_type": "basic",
      "user": "redfishuser",
      "password": "cyAFca7f5i",
      "metrics": {
        "temperature": {
          "path": "/redfish/v1/Chassis/1U/Thermal",
          "plugin": "json",
          "plugin_params": {
            "json_path": "$.Temperatures[0].ReadingCelsius"
          },
          "description": "Temperature",
          "unit": "degC"
        },
        "power": {
          "path": "/redfish/v1/Chassis/1U/Power",
          "plugin": "json",
          "plugin_params": {
            "json_path": "$.PowerControl[0].PowerConsumedWatts"
          },
          "description": "The actual power being consumed by the chassis.",
          "unit": "W"
        }
      }
    }
  }
