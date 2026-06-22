from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from app.core.config import settings
from contextlib import contextmanager
from typing import Generator, Any

_client = None
_callback_handler = None

def get_langfuse() -> Langfuse | None:
    global _client
    # Treat empty strings or placeholders containing "..." as unconfigured
    if not settings.langfuse_public_key or "..." in settings.langfuse_public_key:
        return None
    if _client is None:
        _client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
    return _client

def get_langfuse_callback() -> CallbackHandler | None:
    global _callback_handler
    if not settings.langfuse_public_key or "..." in settings.langfuse_public_key:
        return None
    if _callback_handler is None:
        _callback_handler = CallbackHandler(
            public_key=settings.langfuse_public_key
        )
    return _callback_handler

class DummySpan:
    def update(self, *args, **kwargs):
        pass

@contextmanager
def start_trace_or_span(name: str, as_type: str = "span", **kwargs) -> Generator[Any, None, None]:
    lf = get_langfuse()
    if lf is not None:
        with lf.start_as_current_observation(name=name, as_type=as_type, **kwargs) as span:
            yield span
    else:
        yield DummySpan()
