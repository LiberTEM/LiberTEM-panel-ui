from __future__ import annotations
import time
import itertools
import os
import contextlib
import hashlib
from collections import deque
from typing import TYPE_CHECKING, AsyncIterable, AsyncIterator, Any

if TYPE_CHECKING:
    import hyperspy.api_nogui as hs


@contextlib.contextmanager
def visible_context(*elements):
    for el in elements:
        el.visible = True
    yield
    for el in elements:
        el.visible = False


@contextlib.contextmanager
def loading_context(*elements):
    for el in elements:
        el.loading = True
    yield
    for el in elements:
        el.loading = False


@contextlib.contextmanager
def disabled_context(*elements):
    for el in elements:
        el.disabled = True
    yield
    for el in elements:
        el.disabled = False


def random_hash(characters=4):
    obj = hashlib.sha256()
    obj.update(os.urandom(16))
    return obj.hexdigest()[:characters]


class Timer:
    def __init__(self, indent=0, factor=1000):
        self.indent = indent
        self.factor = factor
        self.ts = time.perf_counter()
        self.tc = self.ts

    def new(self):
        print('---')
        return self

    def update(self, message):
        te = time.perf_counter()
        print(f'{self.indent * 3 * " "}{message: <20}: '
              f'step = {(te - self.tc) * self.factor:>7.2f} ms, '
              f'elapsed = {(te - self.ts) * self.factor:>7.2f} ms')
        self.tc = time.perf_counter()


def recursive_map(nest, to_apply, target, iterate):
    for el in nest:
        if isinstance(el, target):
            to_apply(el)
            continue
        elif isinstance(el, iterate):
            recursive_map(el, to_apply, target, iterate)
        else:
            pass


def recursive_extract(nest, target, iterate, found=None):
    if found is None:
        found = []
    for el in nest:
        if isinstance(el, target):
            found.append(el)
        elif isinstance(el, iterate):
            recursive_extract(el, target, iterate, found=found)
    return found


def pop_from_list(sequence, el, lazy=True):
    try:
        idx = sequence.index(el)
        sequence.pop(idx)
    except ValueError as e:
        if not lazy:
            raise e


def rolling_n_iter(iterable, n):
    fifo = deque(maxlen=n)
    for el in iterable:
        fifo.append(el)
        if len(fifo) == n:
            yield tuple(fifo)


def pairwise(iterable):
    """
    This exists in itertools ??
    """
    return rolling_n_iter(iterable, 2)


def cycle_n(iterable, n=None):
    if n is None:
        yield from itertools.cycle(iterable)
    else:
        saved = []
        for element in iterable:
            yield element
            saved.append(element)
        cycles = 1
        while saved:
            if cycles >= n:
                break
            for element in saved:
                yield element
            cycles += 1


def deduplicate(iterable):
    retain = []
    skip = False

    for el0, el1 in pairwise(iterable):
        if skip:
            skip = (el0 == el1)
            continue
        retain.append(el0)
        skip = (el0 == el1)

    if not skip:
        retain.append(el1)
    return retain


class staticproperty(staticmethod):
    """
    From https://stackoverflow.com/a/69450324
    """
    def __get__(self, *_):
        return self.__func__()


def get_hs() -> hs:
    try:
        import hyperspy.api_nogui as hs
        return hs
    except ImportError:
        return None


async def asyncenumerate(iterable: AsyncIterable,
                         start: int = 0) -> AsyncIterator[tuple[int, Any]]:
    idx = start
    async for item in iterable:
        yield idx, item
        idx += 1


def auto_name_format(name, idx):
    return f'{name}-{idx}'


def increment_name(base, current_names):
    found = tuple(n for n in current_names if n.startswith(base))
    nfound = len(found)
    candidate = auto_name_format(base, nfound)
    while candidate in found:
        nfound += 1
        candidate = auto_name_format(base, nfound)
    return candidate


def unique_name(base, current_names):
    if base not in current_names:
        return base
    return increment_name(base, current_names)


def extract_from_dict(nest, *keys):
    new_nest = {}
    for k, v in nest.items():
        if k in keys:
            new_nest[k] = v
            continue
        if isinstance(v, dict):
            sub = extract_from_dict(v, *keys)
            if sub:
                new_nest[k] = sub
    return new_nest


def get_initial_pos(shape: tuple[int, int]):
    h, w = shape
    cy, cx = h // 2, w // 2
    ri, r = h // 6, w // 4
    return tuple(map(float, (cy, cx))), tuple(map(float, (ri, r))), float(max(h, w)) * 0.5
