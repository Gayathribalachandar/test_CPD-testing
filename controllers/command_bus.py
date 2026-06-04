from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


EventHandler = Callable[[dict[str, Any]], None]
CommandHandler = Callable[[Any], Any]
CommandMiddleware = Callable[[Any, Callable[[], Any]], Any]


@dataclass(slots=True)
class BaseCommand:
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def command_name(self) -> str:
        return self.__class__.__name__


class EventBus:
    def __init__(self):
        self._subscribers: dict[str, list[EventHandler]] = {}

    def subscribe(self, event_name: str, handler: EventHandler) -> Callable[[], None]:
        handlers = self._subscribers.setdefault(str(event_name), [])
        handlers.append(handler)

        def _unsubscribe() -> None:
            current = self._subscribers.get(str(event_name), [])
            try:
                current.remove(handler)
            except ValueError:
                pass

        return _unsubscribe

    def publish(self, event_name: str, payload: dict[str, Any] | None = None) -> None:
        event_payload = dict(payload or {})
        event_payload.setdefault("event_name", str(event_name))
        for handler in list(self._subscribers.get(str(event_name), [])):
            handler(event_payload)


class CommandBus:
    def __init__(self, event_bus: EventBus | None = None):
        self.event_bus = event_bus or EventBus()
        self._handlers: dict[type[Any], CommandHandler] = {}
        self._middleware: list[CommandMiddleware] = []

    def register_handler(self, command_type: type[Any], handler: CommandHandler) -> None:
        self._handlers[command_type] = handler

    def add_middleware(self, middleware: CommandMiddleware) -> None:
        self._middleware.append(middleware)

    def execute(self, command: Any) -> Any:
        handler = self._handlers.get(type(command))
        if handler is None:
            raise ValueError(f"No handler registered for {type(command).__name__}")

        command_name = getattr(command, "command_name", type(command).__name__)
        self.event_bus.publish(
            "command.started",
            {"command": command, "command_name": command_name},
        )

        def _call_handler() -> Any:
            return handler(command)

        pipeline = _call_handler
        for middleware in reversed(self._middleware):
            next_step = pipeline

            def _make_step(
                current_middleware: CommandMiddleware,
                current_next: Callable[[], Any],
            ) -> Callable[[], Any]:
                return lambda: current_middleware(command, current_next)

            pipeline = _make_step(middleware, next_step)

        try:
            result = pipeline()
        except Exception as exc:
            self.event_bus.publish(
                "command.failed",
                {
                    "command": command,
                    "command_name": command_name,
                    "error": exc,
                },
            )
            raise

        self.event_bus.publish(
            "command.completed",
            {
                "command": command,
                "command_name": command_name,
                "result": result,
            },
        )
        return result
