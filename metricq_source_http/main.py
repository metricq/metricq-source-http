import time
import asyncio
import importlib
from queue import Queue
from urllib.parse import urljoin
import logging
import logging.handlers
import click
import click_log
import aiohttp

import metricq
from metricq.logging import get_logger

NaN = float('nan')

logger = get_logger()

click_log.basic_config(logger)
sh = logging.handlers.SysLogHandler()
logger.addHandler(sh)
logger.setLevel('INFO')
logger.handlers[0].formatter = logging.Formatter(
    fmt='%(asctime)s [%(levelname)-8s] [%(name)-20s] %(message)s')


class ConfigError(Exception):
    pass


async def cookie_auth(host, login_data, session):
    """try to login by login page

    Arguments:
        host {str} -- host
        login_data {dict} -- dict with all login information
        session {aiohttp.Session} -- session where is trying to login
    """
    if time.time() - login_data['last_login_try'] > 4.9:
        login_data['last_login_try'] = time.time()
        try:
            response = await session.post(
                '{}{}'.format(host, login_data['login_path']),
                json={
                    'data': [
                        login_data['user'],
                        login_data['password'],
                    ]
                }
            )
            if response.status >= 400:
                logger.error(
                    'Error in cookie auth by {0}{2}, status code: {1}'
                    .format(
                        host,
                        response.status,
                        login_data['login_path'],
                    )
                )
            else:
                login_data['authorized'] = True
        except asyncio.TimeoutError:
            logger.error(
                'Error in cookie auth by {0}{1}, Timeout'
                .format(
                    host,
                    login_data['login_path'],
                )
            )
        except aiohttp.ClientError as e:
            logger.error(
                'Error in cookie auth by {0}{1}, {2}'
                .format(
                    host,
                    login_data['login_path'],
                    e,
                )
            )


async def query_data(metric_name, conf):
    """trying to get the the value from the url.

    Arguments:
        metric_name {str} -- metric name
        conf {dict}       -- all metric information

    Returns:
        tuple -- (metric_name, timestamp of query, queried value(no value == NaN))
    """
    url = urljoin(conf['host_infos']['host_url'], conf['path'])
    value = NaN
    ts = metricq.Timestamp.now()
    if not conf['host_infos']['login_data']['authorized']:
        return metric_name, ts, value
    json_data = {}
    try:
        response = await conf['host_infos']['session'].get(url)
        if response.status >= 400:
            logger.error(
                'Error by {0}, status code: {1}'
                .format(
                    url,
                    response.status,
                )
            )
            if conf['host_infos']['login_data']['login_type'] == 'cookie':
                conf['host_infos']['login_data']['authorized'] = False
        else:
            json_data = await response.json(content_type=None)
    except asyncio.TimeoutError:
        logger.error(
            'Timeout by query data from {0}'
            .format(
                url,
            )
        )
    except aiohttp.ClientError as e:
        logger.error(
            'Error by query data from {0}, {1}'
            .format(
                url,
                e,
            )
        )
    if json_data:
        full_modul_name = 'metricq_source_http.plugins.p_{}'.format(
            conf['plugin']
        )
        if importlib.util.find_spec(full_modul_name):
            plugin = importlib.import_module(full_modul_name)
            try:
                value = plugin.json_parse(json_data, **conf['plugin_params'])
            except Exception as e:
                logger.error(
                    'Error by parse data from {0}, plugin: {1}, Exception: {2}'
                    .format(
                        url,
                        conf['plugin'],
                        e,
                    )
                )
        else:
            logger.error(
                'Error by {0}, plugin not found: {1}'
                .format(
                    url,
                    conf['plugin'],
                )
            )
    return metric_name, ts, value


async def collect_periodically(metric_name, conf, result_queue):
    """loop that collect data for one metric

    Arguments:
        metric_name {str} -- metric_name
        conf {dict} -- config
        result_queue {Queue} -- result_queue
    """
    deadline = time.time() + conf['interval']
    while True:
        while deadline <= time.time():
            logging.warning('missed deadline')
            deadline += conf['interval']
        sleep_var = deadline - time.time()
        await asyncio.sleep(sleep_var)
        deadline += conf['interval']
        if not conf['host_infos']['login_data']['authorized']:
            await cookie_auth(
                conf['host_infos']['host_url'],
                conf['host_infos']['login_data'],
                conf['host_infos']['session'],
            )
        result = await query_data(
            metric_name,
            conf,
        )
        result_queue.put(result)


def make_session(login_data, timeout):
    """make a session

    Arguments:
        login_data {dict} -- dict with all login information
        timeout {int} -- timeout

    Returns:
        aiohttp.Session -- session
    """
    auth = None
    if login_data['login_type'] == 'basic':
        auth = aiohttp.BasicAuth(
            login=login_data['user'],
            password=login_data['password'],
        )
    session = aiohttp.ClientSession(
        auth=auth,
        connector=aiohttp.TCPConnector(ssl=False),
        cookie_jar=aiohttp.CookieJar(unsafe=True),
        timeout=aiohttp.ClientTimeout(total=timeout),
    )
    return session


def check_login_conf(host, conf):
    """check if the conf complete

    Arguments:
        host {str} -- host to which the config belongs
        conf {dict} -- config

    Raises:
        ConfigError: raises if config is not complete

    Returns:
        dict -- extended config
    """
    host_login_infos = {
        'authorized': True,
        'last_login_try': time.time()-5,
    }
    if not 'login_type' in conf:
        raise ConfigError(
            "login_type missing in {}".format(host)
        )
    host_login_infos['login_type'] = conf['login_type']
    if conf['login_type'] != 'none':
        try:
            host_login_infos['user'] = conf['user']
            host_login_infos['password'] = conf['password']
        except KeyError:
            raise ConfigError(
                "user or/and password missing in {}".format(host)
            )
        if conf['login_type'] == 'cookie':
            if not 'login_path' in conf:
                raise ConfigError(
                    "login_path missing in {}".format(host)
                )
            host_login_infos['login_path'] = conf['login_path']
            host_login_infos['authorized'] = False
    return host_login_infos


def make_conf_and_metrics(conf, default_interval, timeout):
    """rebuild config and make the metrics to declaire

    Arguments:
        conf {dict} -- config to rebuild
        default_interval {int} -- interval if no other exists
        timeout {int} -- timeout

    Raises:
        ConfigError: raises if 'path' is not in conf

    Returns:
        tuple -- rebuilded conf, metrics
    """
    metrics = {}
    new_conf = {}
    for host in conf:
        host_login_data = check_login_conf(host, conf[host])
        session = make_session(host_login_data, timeout)
        for metric in conf[host]['metrics']:
            metric_name = '{0}.{1}'.format(
                conf[host]['name'],
                metric,
            )
            interval = conf[host]['metrics'][metric].get(
                'interval',
                default_interval
            )
            metrics[metric_name] = {
                'rate': interval,
            }
            if 'unit' in conf[host]['metrics'][metric]:
                metrics[metric_name]['unit'] = conf[host]['metrics'][metric]['unit']
            if 'description' in conf[host]['metrics'][metric]:
                metrics[metric_name]['description'] = conf[host]['metrics'][metric]['description']

            if 'insecure' in conf[host] and conf[host]['insecure']:
                host_url = '{}{}'.format('http://', host)
            else:
                host_url = '{}{}'.format('https://', host)

            if not 'path' in conf[host]['metrics'][metric]:
                raise ConfigError(
                    "path missing in {}: {}".format(host, metric)
                )
            if not 'plugin' in conf[host]['metrics'][metric]:
                raise ConfigError(
                    "'plugin' missing in {}: {}".format(host, metric)
                )
            new_conf[metric_name] = {
                'path': conf[host]['metrics'][metric]['path'],
                'plugin': conf[host]['metrics'][metric]['plugin'],
                'plugin_params': conf[host]['metrics'][metric].get(
                    'plugin_params',
                    {},
                ),
                'host_infos': {
                    'host_url': host_url,
                    'login_data': host_login_data,
                    'session': session,
                },
                'interval': interval,
            }

    return new_conf, metrics


class HttpSource(metricq.IntervalSource):
    def __init__(self, *args, **kwargs):
        self.period = None
        self.result_queue = Queue()
        logger.info("initializing HttpSource")
        super().__init__(*args, **kwargs)
        watcher = asyncio.FastChildWatcher()
        watcher.attach_loop(self.event_loop)
        asyncio.set_child_watcher(watcher)

    @metricq.rpc_handler('config')
    async def _on_config(self, **config):
        self.period = 1
        new_conf, metrics = make_conf_and_metrics(
            config['hosts'],
            config.get('interval', 1),
            config.get('http_timeout', 5),
        )
        await self.declare_metrics(metrics)
        logger.info("declared {} metrics".format(len(metrics)))
        request_loops = []
        for metric_name, conf in new_conf.items():
            request_loops.append(
                collect_periodically(
                    metric_name,
                    conf,
                    self.result_queue,
                )
            )
        asyncio.gather(*request_loops)  # FIXME: close loops when _on_config is called multiple times

    async def update(self):
        send_metrics = []
        while not self.result_queue.empty():
            metric_name, ts, value = self.result_queue.get()
            send_metrics.append(self[metric_name].send(ts, value))
        if send_metrics:
            ts_before = time.time()
            await asyncio.gather(*send_metrics)
            logger.info("Send took {:.2f} seconds, count: {}".format(
                time.time() - ts_before, len(send_metrics)))


@click.command()
@click.option('--server', default='amqp://localhost/')
@click.option('--token', default='source-http')
@click_log.simple_verbosity_option(logger)
def run(server, token):
    src = HttpSource(token=token, management_url=server)
    src.run()


if __name__ == '__main__':
    run()