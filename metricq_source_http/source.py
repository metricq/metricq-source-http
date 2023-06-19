import asyncio
import importlib
import logging
import logging.handlers
import time
from collections import defaultdict
from contextlib import suppress
from dataclasses import dataclass
from functools import cache
from typing import Any, Callable, Iterable, Optional, Sequence, cast

import aiohttp
import click
import click_log  # type: ignore
import metricq
from hostlist import expand_hostlist  # type: ignore
from metricq.logging import get_logger
from yarl import URL

from .version import __version__

logger = get_logger()

click_log.basic_config(logger)
logger.addHandler(logging.handlers.SysLogHandler())
logger.setLevel("INFO")
logger.handlers[0].formatter = logging.Formatter(
    fmt="%(asctime)s [%(levelname)-8s] [%(name)-20s] %(message)s"
)


class ConfigError(Exception):
    pass


@cache
def load_plugin(name: str) -> Callable[..., float]:
    try:
        module = importlib.import_module(f"metricq_source_http.plugins.{name}")
    except ImportError as e:
        raise ConfigError(f"Could not load plugin {name}: {e}")
    try:
        return cast(Callable[..., float], module.response_parse)
    except AttributeError:
        raise ConfigError(f"Plugin {name} has no response_parse function")


def _extract_interval(**config: Any) -> Optional[metricq.Timedelta]:
    """
    To allow interval, rate or maybe period, and other types in the future,
    we have a separate function here.
    """
    with suppress(KeyError):
        interval = config["interval"]
        if isinstance(interval, (int, float)):
            return metricq.Timedelta.from_s(interval)
        else:
            assert isinstance(interval, str)
            return metricq.Timedelta.from_string(interval)
    return None


class AuthorizationManager:
    _needs_login = False

    @staticmethod
    def create(
        login_type: Optional[str] = None, **kwargs: Any
    ) -> "AuthorizationManager":
        if login_type is None or login_type == "none":
            return AuthorizationManager()
        if login_type == "cookie":
            return CookieAuthorizationManager(**kwargs)
        if login_type == "basic":
            return BasicAuthorizationManager(**kwargs)
        raise ConfigError(f"login_type {login_type} not supported")

    @property
    def session_params(self) -> dict[str, Any]:
        return {}

    async def authorize_session(self, session: aiohttp.ClientSession) -> bool:
        """
        Authorize the session if that is somehow possible.
        This only returns ``True`` if something was done, and it was successful.
        ``True`` therefore indicates, that the caller can retry if there
        previously was an error. If ``False`` is returned, you need not retry.
        """
        return False


class BasicAuthorizationManager(AuthorizationManager):
    def __init__(self, *, user: str, password: str, **kwargs: Any):
        self.user = user
        self.password = password

    @property
    def session_params(self) -> dict[str, Any]:
        return {"auth": aiohttp.BasicAuth(self.user, self.password)}


class CookieAuthorizationManager(AuthorizationManager):
    def __init__(self, *, login_path: str, user: str, password: str, **kwargs: Any):
        self.login_path = login_path
        self.user = user
        self.password = password

    async def authorize_session(self, session: aiohttp.ClientSession) -> bool:
        try:
            response = await session.post(
                self.login_path,
                json={
                    "data": [self.user, self.password],
                },
            )
            response.raise_for_status()
            return True
        except (asyncio.TimeoutError, aiohttp.ClientError) as e:
            logger.error(f"Error in cookie auth at {self.login_path}: {e}")
            return False

    @property
    def session_params(self) -> dict[str, Any]:
        # `unsafe=True` is needed to allow cookies from URLs with ip addresses.
        # This is fine, we don't wildly follow links and only access the configured host
        return {"cookie_jar": aiohttp.CookieJar(unsafe=True)}


@dataclass(frozen=True)
class _MetricGroupKey:
    interval: metricq.Timedelta
    path: str


class Metric:
    def __init__(
        self,
        name: str,
        host: "Host",
        *,
        path: str,
        plugin: str,
        plugin_params: dict[str, Any] = {},
        description: str = "",
        unit: Optional[str] = None,
        chunk_size: Optional[int] = None,
        **kwargs: Any,
    ):
        self._source_metric = host.source[name]
        if chunk_size is not None:
            self._source_metric.chunk_size = chunk_size
        self.description = description
        if host.description:
            self.description = f"{host.description} {self.description}"

        interval = _extract_interval(**kwargs)
        if interval is None:
            interval = host.source.default_interval
            if interval is None:
                raise ConfigError(f"interval missing in {name}")
        self.interval: metricq.Timedelta = interval
        if not path.startswith("/"):
            logger.warning(
                f"Path '{path}' for {name} should start with '/', adding it."
            )
            path = f"/{path}"
        self.path = path
        self.plugin_parse = load_plugin(plugin)
        self.plugin_params = plugin_params
        self.unit = unit

    @property
    def name(self) -> str:
        return self._source_metric.id

    @property
    def metadata(self) -> metricq.JsonDict:
        metadata = {
            "description": self.description,
            "rate": 1 / self.interval.s,
            "interval": self.interval.s,
        }
        if self.unit:
            metadata["unit"] = self.unit
        return metadata

    @property
    def group_key(self) -> _MetricGroupKey:
        return _MetricGroupKey(interval=self.interval, path=self.path)

    async def update(self, timestamp: metricq.Timestamp, text: str) -> None:
        try:
            value = self.plugin_parse(text, **self.plugin_params)
        except Exception as e:
            logger.error(f"Error in plugin parse for {self.name}: {e}")
            return
        try:
            await self._source_metric.send(timestamp, value)
        except Exception as e:
            logger.error(f"Error in metric send for {self.name}: {e}")


class MetricGroup:
    """
    Represents a set of metrics
    - same host (implicitly)
    - same interval (by key in host)
    - same path (by key in host)
    """

    _metrics: list[Metric]
    _key: Optional[_MetricGroupKey] = None

    def __init__(self, host: "Host") -> None:
        self._host = host
        self._metrics = []

    def add(self, metric: Metric) -> None:
        self._metrics.append(metric)
        if self._key is None:
            self._key = metric.group_key
        assert self._key == metric.group_key

    @property
    def _interval(self) -> metricq.Timedelta:
        assert self._key is not None
        return self._key.interval

    @property
    def _path(self) -> str:
        assert self._key is not None
        return self._key.path

    @property
    def metadata(self) -> dict[str, metricq.MetadataDict]:
        return {metric.name: metric.metadata for metric in self._metrics}

    async def task(
        self,
        stop_future: asyncio.Future[None],
        authorization_manager: AuthorizationManager,
        session: aiohttp.ClientSession,
    ) -> None:
        # Similar code as to metricq.IntervalSource.task, but for individual MetricGroups
        deadline = metricq.Timestamp.now()
        deadline -= metricq.Timedelta(deadline.posix_ns % self._interval.ns)  #
        # Align
        # deadlines to
        # the
        # interval
        while True:
            try:
                await self._update(session, authorization_manager)
            except Exception as e:
                logger.error(f"Error in MetricGroup._update: {e} ({type(e)}")
                # Just retry indefinitely

            deadline += self._interval

            now = metricq.Timestamp.now()
            while now >= deadline:
                logger.warning("Missed deadline {}, it is now {}", deadline, now)
                deadline += self._interval

            timeout = deadline - now
            done, pending = await asyncio.wait(
                (asyncio.create_task(asyncio.sleep(timeout.s)), stop_future),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if stop_future in done:
                for task in pending:  # cancel pending sleep task
                    task.cancel()
                stop_future.result()  # potentially raise exceptions
                return

    async def _update(
        self,
        session: aiohttp.ClientSession,
        authorization_manager: AuthorizationManager,
    ) -> None:
        timestamp = metricq.Timestamp.now()
        try:
            response = await session.get(self._path)
            if not response.ok and authorization_manager.authorize_session(session):
                logger.debug(f"Re-trying request to {self._path} with authorization")
                response = await session.get(self._path)
            response.raise_for_status()
            text = await response.text()
        except aiohttp.ClientError as e:
            logger.error(f"Error in request to {self._host.base_url}{self._path}: {e}")
            return

        duration = metricq.Timestamp.now() - timestamp
        logger.debug(
            f"Request '{self._host.base_url}{self._path}' finished successfully in {duration}"
        )

        await asyncio.gather(
            *(metric.update(timestamp, text) for metric in self._metrics)
        )


class Host:
    def __init__(
        self,
        source: "HttpSource",
        *,
        host: str,
        name: str,
        metrics: dict[str, Any],
        description: str = "",
        insecure: bool = False,
        **kwargs: Any,
    ):
        self.source = source
        self._scheme = "http" if insecure else "https"
        self._host = host
        self._metric_prefix = name
        self.description = description

        self._groups: defaultdict[_MetricGroupKey, MetricGroup] = defaultdict(
            lambda: MetricGroup(self)
        )
        self._add_metrics(metrics)
        self._authorization_manager = AuthorizationManager.create(**kwargs)

    def _add_metrics(self, metrics: dict[str, Any]) -> None:
        for metric_suffix, metric_data in metrics.items():
            metric_name = f"{self._metric_prefix}.{metric_suffix}"
            metric = Metric(
                metric_name,
                self,
                **metric_data,
            )
            self._groups[metric.group_key].add(metric)

    @staticmethod
    def _parse_hosts(hosts: str | list[str]) -> list[str]:
        if isinstance(hosts, str):
            return cast(list[str], expand_hostlist(hosts))
        assert isinstance(hosts, list)
        assert all(isinstance(host, str) for host in hosts)
        return hosts

    @classmethod
    def _create_from_host_config(
        cls,
        source: "HttpSource",
        *,
        hosts: str | list[str],
        names: str | list[str],
        **kwargs: Any,
    ) -> Iterable["Host"]:
        hosts = cls._parse_hosts(hosts)
        names = cls._parse_hosts(names)
        if len(hosts) != len(names):
            raise ConfigError("Number of names and hosts differ")
        for host, name in zip(hosts, names):
            yield Host(source=source, host=host, name=name, **kwargs)

    @classmethod
    def create_from_host_configs(
        cls, source: "HttpSource", host_configs: Sequence[dict[str, Any]]
    ) -> Iterable["Host"]:
        for host_config in host_configs:
            yield from cls._create_from_host_config(source, **host_config)

    @property
    def metadata(self) -> dict[str, metricq.MetadataDict]:
        return {
            metric: metadata
            for group in self._groups.values()
            for metric, metadata in group.metadata.items()
        }

    @property
    def base_url(self) -> URL:
        return URL.build(scheme=self._scheme, host=self._host)

    async def task(self, stop_future: asyncio.Future[None]) -> None:
        async with aiohttp.ClientSession(
            base_url=self.base_url,
            timeout=aiohttp.ClientTimeout(total=self.source.http_timeout),
            **self._authorization_manager.session_params,
        ) as session:
            await self._authorization_manager.authorize_session(session)
            await asyncio.gather(
                *[
                    group.task(stop_future, self._authorization_manager, session)
                    for group in self._groups.values()
                ]
            )


class HttpSource(metricq.Source):
    default_interval: Optional[metricq.Timedelta] = None
    hosts: Optional[list[Host]] = None
    _host_task_stop_future: Optional[asyncio.Future[None]] = None
    _host_task: Optional[asyncio.Task[None]] = None

    @metricq.rpc_handler("config")
    async def _on_config(
        self,
        *,
        hosts: list[dict[str, Any]],
        http_timeout: int | float = 5,
        **config: Any,
    ) -> None:
        self.default_interval = _extract_interval(**config)
        self.http_timeout = http_timeout

        if self.hosts is not None:
            await self._stop_host_tasks()

        self.hosts = list(Host.create_from_host_configs(self, hosts))

        await self.declare_metrics(
            {
                metric: metadata
                for host in self.hosts
                for metric, metadata in host.metadata.items()
            }
        )

        self._create_host_tasks()

    async def _stop_host_tasks(self) -> None:
        assert self._host_task_stop_future is not None
        assert self._host_task is not None
        self._host_task_stop_future.set_result(None)
        try:
            await asyncio.wait_for(self._host_task, timeout=30)
        except asyncio.TimeoutError:
            # wait_for also cancels the task
            logger.error("Host tasks did not stop in time")
        self.hosts = None
        self._host_task_stop_future = None
        self._host_task = None

    def _create_host_tasks(self) -> None:
        assert self.hosts is not None
        assert self._host_task_stop_future is None
        assert self._host_task is None
        self._host_task_stop_future = asyncio.Future()
        self._host_task = asyncio.create_task(self._run_host_tasks())

    async def _run_host_tasks(self) -> None:
        assert self._host_task_stop_future is not None
        assert self.hosts is not None
        await asyncio.gather(
            *(host.task(self._host_task_stop_future) for host in self.hosts)
        )

    async def task(self) -> None:
        """
        Just wait for the global task_stop_future and propagate it to the host tasks.
        """
        assert self.task_stop_future is not None
        await self.task_stop_future
        await self._stop_host_tasks()


@click.command()
@click.option("--server", default="amqp://localhost/")
@click.option("--token", default="source-http")
@click_log.simple_verbosity_option(logger)  # type: ignore
def main(server: str, token: str) -> None:
    src = HttpSource(token=token, url=server)
    src.run()


if __name__ == "__main__":
    main()
