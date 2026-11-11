"""A small reentrant read-write lock: many concurrent searches, one writer at a time."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Iterator


class RWLock:
    """``with lock:`` takes the (reentrant) write lock; ``with lock.read():`` a shared read lock.

    Writers are preferred, so a stream of searches can't starve a write. A thread holding
    the write lock may also take read locks. Taking the write lock while holding only a
    read lock would deadlock, so callers never do that.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition(threading.Lock())
        self._readers = 0
        self._writer: int | None = None
        self._depth = 0
        self._waiting_writers = 0

    @contextmanager
    def read(self) -> Iterator[None]:
        me = threading.get_ident()
        if self._writer == me:  # already exclusive
            yield
            return
        with self._cond:
            while self._writer is not None or self._waiting_writers:
                self._cond.wait()
            self._readers += 1
        try:
            yield
        finally:
            with self._cond:
                self._readers -= 1
                if not self._readers:
                    self._cond.notify_all()

    def acquire(self) -> None:
        me = threading.get_ident()
        with self._cond:
            if self._writer == me:
                self._depth += 1
                return
            self._waiting_writers += 1
            while self._writer is not None or self._readers:
                self._cond.wait()
            self._waiting_writers -= 1
            self._writer, self._depth = me, 1

    def release(self) -> None:
        with self._cond:
            self._depth -= 1
            if not self._depth:
                self._writer = None
                self._cond.notify_all()

    def __enter__(self) -> RWLock:
        self.acquire()
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()
