from setuptools import setup

setup(
    name="metricq_source_http",
    version="0.1",
    author="TU Dresden",
    python_requires=">=3.10",
    packages=[
        "metricq_source_http.source",
        "metricq_source_http.plugins",
    ],
    scripts=[],
    entry_points="""
      [console_scripts]
      metricq-source-http=metricq_source_http.source:run
      """,
    install_requires=[
        "click",
        "click_log",
        "metricq ~= 5.1",
        "aiohttp",
        "jsonpath-rw",
        "python-hostlist",
    ],
)
