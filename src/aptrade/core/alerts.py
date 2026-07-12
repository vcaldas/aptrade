import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from aptrade.telegram_bot import get_telegram_bot, telegram_notifications_enabled

from .config import settings
from .logger import get_logger


def _parse_event_list(raw: str) -> set[str]:
    return {item.strip().lower() for item in raw.split(",") if item.strip()}


@dataclass(frozen=True)
class AlertEvent:
    name: str
    message: str
    level: int = logging.INFO
    notify_telegram: bool = False


class AlertPublisher:
    """Explicit business-event publisher with optional Telegram delivery."""

    def __init__(
        self,
        telegram_enabled: bool,
        telegram_events: Iterable[str],
    ) -> None:
        self._logger = get_logger()
        self._telegram_enabled = telegram_enabled
        self._telegram_events = {name.lower() for name in telegram_events}
        self._telegram_bot = None

    @staticmethod
    def _default_telegram_text(event: AlertEvent) -> str:
        return f"[{event.name}] {event.message}"

    def publish(
        self,
        event: AlertEvent,
        telegram_formatter: Callable[[AlertEvent], str] | None = None,
        **log_kwargs: Any,
    ) -> None:
        event_name = event.name.lower()
        self._logger.log(
            event.level, "[%s] %s", event.name, event.message, **log_kwargs
        )

        if (
            not event.notify_telegram
            or not self._telegram_enabled
            or not telegram_notifications_enabled()
        ):
            return

        if self._telegram_events and event_name not in self._telegram_events:
            return

        if self._telegram_bot is None:
            self._telegram_bot = get_telegram_bot()

        if self._telegram_bot is None:
            self._logger.warning(
                "Telegram alert requested but bot is not configured; event=%s",
                event.name,
            )
            return

        formatter = telegram_formatter or self._default_telegram_text
        try:
            telegram_text = formatter(event)
        except Exception as exc:
            self._logger.warning(
                "Telegram formatter failed for event=%s, using default format: %s",
                event.name,
                exc,
            )
            telegram_text = self._default_telegram_text(event)

        self._telegram_bot.send_message("ALERT", telegram_text)

    @staticmethod
    def _render_message(msg: Any, fmt_args: tuple[Any, ...]) -> str:
        if not fmt_args:
            return str(msg)
        try:
            return str(msg) % fmt_args
        except Exception:
            return f"{msg} {' '.join(str(a) for a in fmt_args)}"

    def _emit_level(self, level: int, *args: Any, **kwargs: Any) -> None:
        notify_telegram = bool(kwargs.pop("notify_telegram", False))
        telegram_formatter = kwargs.pop("telegram_formatter", None)
        event_name = kwargs.pop("name", None)
        event_message = kwargs.pop("message", None)

        if args:
            self._logger.log(level, *args, **kwargs)
            rendered_message = self._render_message(args[0], tuple(args[1:]))
        else:
            message_text = "" if event_message is None else str(event_message)
            self._logger.log(level, "%s", message_text, **kwargs)
            rendered_message = message_text

        if not notify_telegram:
            return

        self.publish(
            AlertEvent(
                name=str(event_name or logging.getLevelName(level).lower()),
                message=rendered_message,
                level=level,
                notify_telegram=True,
            ),
            telegram_formatter=telegram_formatter,
        )

    def emit(
        self,
        *,
        name: str,
        message: str,
        level: int = logging.INFO,
        notify_telegram: bool = False,
        telegram_formatter: Callable[[AlertEvent], str] | None = None,
        **log_kwargs: Any,
    ) -> None:
        self._emit_level(
            level,
            message,
            name=name,
            notify_telegram=notify_telegram,
            telegram_formatter=telegram_formatter,
            **log_kwargs,
        )

    def debug(self, *args: Any, **kwargs: Any) -> None:
        self._emit_level(logging.DEBUG, *args, **kwargs)

    def info(self, *args: Any, **kwargs: Any) -> None:
        self._emit_level(logging.INFO, *args, **kwargs)

    def warning(self, *args: Any, **kwargs: Any) -> None:
        self._emit_level(logging.WARNING, *args, **kwargs)

    def error(self, *args: Any, **kwargs: Any) -> None:
        self._emit_level(logging.ERROR, *args, **kwargs)

    def critical(self, *args: Any, **kwargs: Any) -> None:
        self._emit_level(logging.CRITICAL, *args, **kwargs)

    def exception(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("exc_info", True)
        self._emit_level(logging.ERROR, *args, **kwargs)


_alert_publisher: AlertPublisher | None = None


def get_alert_publisher() -> AlertPublisher:
    global _alert_publisher

    if _alert_publisher is None:
        configured_events = _parse_event_list(settings.ALERTS_TELEGRAM_EVENTS)
        _alert_publisher = AlertPublisher(
            telegram_enabled=settings.ALERTS_TELEGRAM_ENABLED,
            telegram_events=configured_events,
        )

    return _alert_publisher
