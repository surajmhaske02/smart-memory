import pytest

from smart_memory import Memory


class FakeClock:
    def __init__(self, start: float = 1_700_000_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, days: float = 0.0, seconds: float = 0.0) -> None:
        self.now += days * 86_400 + seconds


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def mem(clock):
    with Memory(clock=clock) as m:
        yield m


@pytest.fixture
def alice(mem):
    return mem.scope(user_id="alice", agent_id="support", thread_id="t1")
