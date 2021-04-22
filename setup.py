from setuptools import setup

setup(name='metricq_source_http',
      version='0.1',
      author='TU Dresden',
      python_requires=">=3.6",
      packages=['metricq_source_http.source', 'metricq_source_http.plugin_json', 'metricq_source_http.plugin_openbmc'],
      scripts=[],
      entry_points='''
      [console_scripts]
      metricq-source-http=metricq_source_http.source:run
      ''',
    install_requires=[
        "aiomonitor",
        "click",
        "click_log",
        "metricq ~= 2.0",
        "aiohttp",
        "jsonpath-rw",
        "python-hostlist",
    ],
)
