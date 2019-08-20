from setuptools import setup, find_packages

setup(name='metricq_source_http',
      version='0.1',
      author='TU Dresden',
      python_requires=">=3.6",
      packages=find_packages(),
      scripts=[],
      entry_points='''
      [console_scripts]
      metricq-source-http=metricq_source_http:run
      ''',
      install_requires=['aiomonitor', 'click', 'click_log', 'metricq', 'aiohttp', 'jsonpath-rw'])
