import asyncio
import importlib
import logging
import logging.handlers
import time
from collections import defaultdict
from contextlib import nullcontext
from queue import Queue
from urllib.parse import urljoin

import aiohttp
import click
import click_log
import hostlist
import metricq
from expiringdict import ExpiringDict
from metricq import Timedelta
from metricq.logging import get_logger

NaN = float("nan")
LOADED_PLUGINS = {}

logger = get_logger()

click_log.basic_config(logger)
sh = logging.handlers.SysLogHandler()
logger.addHandler(sh)
logger.setLevel("INFO")
logger.handlers[0].formatter = logging.Formatter(
    fmt="%(asctime)s [%(levelname)-8s] [%(name)-20s] %(message)s"
)

cache = ExpiringDict(max_len=5, max_age_seconds=5)
cache_locks = defaultdict(asyncio.Lock)


class ConfigError(Exception):
    pass


def hostlist_sanity_check():
    expected = ["a02", "a03", "b02", "b01", "c09", "c07", "a01"]
    hostlist_str = "a[02-03],b[02,01],c09,c07,a01"
    if not hostlist.expand_hostlist(hostlist_str) == expected:
        raise Exception("hostlist sort output!")


async def cookie_auth(host, login_data, session):
    """try to login by login page

    Arguments:
        host {str} -- host
        login_data {dict} -- dict with all login information
        session {aiohttp.Session} -- session where is trying to login
    """
    if time.time() - login_data["last_login_try"] > 4.9:
        login_data["last_login_try"] = time.time()
        try:
            response = await session.post(
                "{}{}".format(host, login_data["login_path"]),
                json={
                    "data": [
                        login_data["user"],
                        login_data["password"],
                    ]
                },
            )
            if response.status >= 400:
                logger.error(
                    "Error in cookie auth by {0}{2}, status code: {1}".format(
                        host,
                        response.status,
                        login_data["login_path"],
                    )
                )
            else:
                login_data["authorized"] = True
        except asyncio.TimeoutError:
            logger.error(
                "Error in cookie auth by {0}{1}, Timeout".format(
                    host,
                    login_data["login_path"],
                )
            )
        except aiohttp.ClientError as e:
            logger.error(
                "Error in cookie auth by {0}{1}, {2}".format(
                    host,
                    login_data["login_path"],
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
    url = urljoin(conf["host_infos"]["host_url"], conf["path"])
    value = NaN
    ts = metricq.Timestamp.now()
    if not conf["host_infos"]["login_data"]["authorized"]:
        return metric_name, ts, value
    data = None

    if conf["use_cache"]:
        locked_cache = cache_locks[url]
    else:
        locked_cache = nullcontext()

    async with locked_cache:
        if conf["use_cache"] and (url in cache):
            data = cache[url]
        else:
            try:
                response = await conf["host_infos"]["session"].get(url)
                if response.status >= 400:
                    logger.error(
                        "Error by {0}, status code: {1}".format(
                            url,
                            response.status,
                        )
                    )

                    if conf["host_infos"]["login_data"]["login_type"] == "cookie":
                        conf["host_infos"]["login_data"]["authorized"] = False

                else:
                    data = await response.text()
                    if conf["use_cache"]:
                        cache[url] = data

            except asyncio.TimeoutError:
                logger.error(
                    "Timeout by query data from {0}".format(
                        url,
                    )
                )
            except aiohttp.ClientError as e:
                logger.error(
                    "Error by query data from {0}, {1}".format(
                        url,
                        e,
                    )
                )

    if data:
        if not conf["plugin"] in LOADED_PLUGINS:
            full_module_name = "metricq_source_http.plugin_{}".format(
                conf["plugin"],
            )
            if importlib.util.find_spec(full_module_name):
                LOADED_PLUGINS[conf["plugin"]] = importlib.import_module(
                    full_module_name
                )
            else:
                logger.error(
                    "Error by {0}, plugin not found: {1}".format(
                        url,
                        conf["plugin"],
                    )
                )

        if conf["plugin"] in LOADED_PLUGINS:
            try:
                value = LOADED_PLUGINS[conf["plugin"]].response_parse(
                    data,
                    **conf["plugin_params"],
                )
            except Exception as e:
                logger.error(
                    "Error by parse data from {0}, plugin: {1}, Exception: {2}".format(
                        url,
                        conf["plugin"],
                        e,
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
    deadline = time.time() + conf["interval"]
    while True:
        if not conf["host_infos"]["login_data"]["authorized"]:
            await cookie_auth(
                conf["host_infos"]["host_url"],
                conf["host_infos"]["login_data"],
                conf["host_infos"]["session"],
            )
        result = await query_data(
            metric_name,
            conf,
        )
        result_queue.put(result)
        while deadline <= time.time():
            logging.warning("missed deadline")
            deadline += conf["interval"]
        sleep_var = deadline - time.time()
        await asyncio.sleep(sleep_var)
        deadline += conf["interval"]


def make_session(login_data, timeout):
    """make a session

    Arguments:
        login_data {dict} -- dict with all login information
        timeout {int} -- timeout

    Returns:
        aiohttp.Session -- session
    """
    auth = None
    if login_data["login_type"] == "basic":
        auth = aiohttp.BasicAuth(
            login=login_data["user"],
            password=login_data["password"],
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
        "authorized": True,
        "last_login_try": time.time() - 5,
    }
    if not "login_type" in conf:
        raise ConfigError("login_type missing in {}".format(host))
    host_login_infos["login_type"] = conf["login_type"]
    if conf["login_type"] != "none":
        try:
            host_login_infos["user"] = conf["user"]
            host_login_infos["password"] = conf["password"]
        except KeyError:
            raise ConfigError("user or/and password missing in {}".format(host))
        if conf["login_type"] == "cookie":
            if not "login_path" in conf:
                raise ConfigError("login_path missing in {}".format(host))
            host_login_infos["login_path"] = conf["login_path"]
            host_login_infos["authorized"] = False
    return host_login_infos


def get_hostlist(obj):
    """check is str than parse to list with hostlist

    Arguments:
        obj {str/list} -- obj to check

    Returns:
        list -- list
    """
    if type(obj) is str:
        return hostlist.expand_hostlist(obj)
    else:
        return obj


def make_conf_and_metrics(conf, default_interval, timeout):
    """rebuild config and make the metrics to declare

    Arguments:
        conf {list<dict>} -- configs to rebuild
        default_interval {int} -- interval if no other exists
        timeout {int} -- timeout

    Raises:
        ConfigError: raises if 'path' is not in conf

    Returns:
        tuple -- rebuilt conf, metrics
    """
    metrics = {}
    new_conf = {}
    for host_data in conf:
        use_cache = host_data.get("use_cache", False)
        hosts = get_hostlist(host_data["hosts"])
        host_names = get_hostlist(host_data["names"])
        if len(hosts) == len(host_names):
            for host, host_name in zip(hosts, host_names):
                host_login_data = check_login_conf(host, host_data)
                session = make_session(host_login_data, timeout)
                for metric, metric_data in host_data["metrics"].items():
                    metric_name = "{0}.{1}".format(
                        host_name,
                        metric,
                    )
                    interval = metric_data.get("interval", default_interval)
                    metrics[metric_name] = {
                        "rate": 1.0 / interval,
                    }
                    if "unit" in metric_data:
                        metrics[metric_name]["unit"] = metric_data["unit"]
                    else:
                        logger.warning(
                            "no unit given in {}".format(
                                metric_name,
                            )
                        )

                    if "description" in host_data:
                        if "description" in metric_data:
                            metrics[metric_name]["description"] = "{0} {1}".format(
                                host_data["description"],
                                metric_data["description"],
                            )
                        else:
                            logger.warning(
                                "host description given but no metric description in {}".format(
                                    metric_name,
                                )
                            )
                    else:
                        if "description" in metric_data:
                            metrics[metric_name]["description"] = metric_data[
                                "description"
                            ]
                        else:
                            logger.warning(
                                "no description given in {}".format(
                                    metric_name,
                                )
                            )

                    if "insecure" in host_data and host_data["insecure"]:
                        host_url = "{}{}".format("http://", host)
                    else:
                        host_url = "{}{}".format("https://", host)

                    if not "path" in metric_data:
                        raise ConfigError("path missing in {}: {}".format(host, metric))
                    if not "plugin" in metric_data:
                        raise ConfigError(
                            "'plugin' missing in {}: {}".format(host, metric)
                        )
                    new_conf[metric_name] = {
                        "use_cache": use_cache,
                        "path": metric_data["path"],
                        "plugin": metric_data["plugin"],
                        "plugin_params": metric_data.get(
                            "plugin_params",
                            {},
                        ),
                        "host_infos": {
                            "host_url": host_url,
                            "login_data": host_login_data,
                            "session": session,
                        },
                        "interval": interval,
                    }
        else:
            raise ConfigError(
                "number of names and hosts different in {} ".format(host_data)
            )
    return new_conf, metrics


class HttpSource(metricq.IntervalSource):
    def __init__(self, *args, **kwargs):
        hostlist_sanity_check()
        self.result_queue = Queue()
        logger.info("initializing HttpSource")
        super().__init__(*args, **kwargs)

    async def connect(self):
        watcher = asyncio.FastChildWatcher()
        watcher.attach_loop(self._event_loop)
        asyncio.set_child_watcher(watcher)

        await super().connect()

    @metricq.rpc_handler("config")
    async def _on_config(self, **config):
        self.period = Timedelta.from_s(1)
        new_conf, metrics = make_conf_and_metrics(
            config["hosts"],
            config.get("interval", 1),
            config.get("http_timeout", 5),
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
        asyncio.gather(
            *request_loops
        )  # FIXME: close loops when _on_config is called multiple times

    async def update(self):
        send_metric_count = 0
        while not self.result_queue.empty():
            metric_name, ts, value = self.result_queue.get()
            if isinstance(value, (int, float)):
                self[metric_name].append(ts, value)
                send_metric_count += 1
            elif value is not None:
                logger.error(
                    f"Value ({value}) for {metric_name} is of type {type(value)}. Can't send via MetricQ!"
                )
        ts_before = time.time()
        try:
            await self.flush()
        except Exception as e:
            logger.error("Exception in send: {}".format(str(e)))
        logger.info(
            "Send took {:.2f} seconds, count: {}".format(
                time.time() - ts_before, send_metric_count
            ),
        )


@click.command()
@click.option("--server", default="amqp://localhost/")
@click.option("--token", default="source-http")
@click_log.simple_verbosity_option(logger)
def run(server, token):
    src = HttpSource(token=token, management_url=server)
    src.run()


if __name__ == "__main__":
    run()
