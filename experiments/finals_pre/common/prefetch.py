"""Overlap day packing with GPU compute.

Profiling on the E9 batch: with six concurrent trainers the GPU sat at 47%
utilisation -- each process alternates pack(CPU) then step(GPU) serially, so the
card idles through every pack. This runs the packing one day ahead on a worker
thread.

The RNG is NOT touched: every day's stock sample is drawn up front, in the same
order the serial loop would draw it, and only the packing is moved off the
critical path. Same seed therefore yields the same days, the same stocks and the
same gradients -- the speedup is pure scheduling.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable, Iterable, Iterator
from typing import Any


def prefetch_days(
    plan: Iterable[tuple[Any, tuple[str, ...], Any]],
    loader: Callable[..., Any],
    store,
    max_minutes: int,
    depth: int = 2,
) -> Iterator[tuple[Any, tuple[str, ...], Any, Any]]:
    """Yield (day, instruments, target, batch), packing `depth` days ahead.

    `plan` must already be materialised in the exact order the serial loop would
    visit it, with the per-day stock sample already drawn.
    """
    plan = list(plan)
    out: queue.Queue = queue.Queue(maxsize=depth)
    error: list[Exception] = []

    def produce() -> None:
        try:
            for day, instruments, target in plan:
                batch = loader(store, day, instruments, max_minutes=max_minutes)
                out.put((day, instruments, target, batch))
        except Exception as exc:  # noqa: BLE001 - a producer thread must catch
            # everything, otherwise a loader error hangs the consumer forever;
            # it is re-raised on the consumer side below.
            error.append(exc)
        finally:
            out.put(None)

    thread = threading.Thread(target=produce, name="day-prefetch", daemon=True)
    thread.start()
    while True:
        item = out.get()
        if item is None:
            break
        yield item
    thread.join()
    if error:
        raise error[0]
