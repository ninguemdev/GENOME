"""Telemetry in Parquet: the ``ticks`` and ``events`` tables (ARQUITETURA §11.1, §11.5, ADR 0009).

A Parquet file without its footer cannot be read, so a run writes numbered parts and
joins them only when it ends. A checkpoint records how many parts count, which lets
a resumed run drop whatever came after it and end with the file of the direct run.
"""

import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq

from genome.core.config import LabConfig
from genome.core.state import NO_ID, CreatureState
from genome.core.state_hash import NamedArray
from genome.core.tick import Simulation, TickReport

IntArray = npt.NDArray[np.int64]

TICKS_FILE = "ticks.parquet"
EVENTS_FILE = "events.parquet"
PARTS_SUFFIX = ".parts"
PART_DIGITS = 6
#: The stack shown of a plant is ⌈b⌉, from 1 to 3 (§4.1).
STACKS = (1, 2, 3)
BIRTH = "birth"
DEATH = "death"
#: Clonal births are the only ones before F3 (ADR 0004, ADR 0009).
ASEXUAL = "asexual"

TICKS_SCHEMA = pa.schema(
    [
        ("tick", pa.int64()),
        ("population", pa.int64()),
        ("births", pa.int64()),
        ("deaths", pa.int64()),
        ("plants_total", pa.int64()),
        ("plants_stack1", pa.int64()),
        ("plants_stack2", pa.int64()),
        ("plants_stack3", pa.int64()),
        ("biomass_total", pa.float64()),
        ("water_lakes", pa.float64()),
        ("water_puddles", pa.float64()),
        ("light", pa.float64()),
        ("rain", pa.bool_()),
        ("humidity", pa.float64()),
        ("mean_energy", pa.float64()),
        ("mean_hydration", pa.float64()),
        ("species_count", pa.int64()),
        ("energy_ledger_error", pa.float64()),
        ("water_ledger_error", pa.float64()),
    ]
)

EVENTS_SCHEMA = pa.schema(
    [
        ("tick", pa.int64()),
        ("event", pa.string()),
        ("uid", pa.int64()),
        ("parent_a", pa.int64()),
        ("parent_b", pa.int64()),
        ("founder_id", pa.int64()),
        ("reproduction", pa.string()),
        ("death_cause", pa.string()),
        ("age", pa.int64()),
        ("mass", pa.float64()),
        ("v_max", pa.float64()),
        ("color", pa.float64()),
        ("species_id", pa.int64()),
    ]
)


class BlockWriter:
    """Accumulates rows and writes them in blocks; one final file at the end (§11.1, §11.5)."""

    def __init__(self, path: Path, schema: pa.Schema, flush_rows: int) -> None:
        """Prepare the writer; nothing touches the disk before the first block."""
        self._path = path
        self._parts = path.with_name(path.name + PARTS_SUFFIX)
        self._schema = schema
        self._flush_rows = flush_rows
        self._pending: list[pa.Table] = []
        self._pending_rows = 0
        self.parts_written = 0

    def add(self, table: pa.Table) -> None:
        """Keep the rows, and write a block once ``flush_rows`` accumulated (§11.5)."""
        self._pending.append(table)
        self._pending_rows += table.num_rows
        if self._pending_rows >= self._flush_rows:
            self.flush()

    def flush(self) -> None:
        """Write what accumulated as the next part, if anything did."""
        if not self._pending_rows:
            return
        self._parts.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.concat_tables(self._pending), self._part(self.parts_written))
        self.parts_written += 1
        self._pending = []
        self._pending_rows = 0

    def require_parts(self, parts: int) -> None:
        """Check that the first ``parts`` parts exist, by name, not just by count (ADR 0009).

        Raises:
            ValueError: If any of them is missing, as after the run was joined.
        """
        found = sum(self._part(index).exists() for index in range(parts))
        if found < parts:
            raise ValueError(f"{self._parts}: expected {parts} parts, found {found}")

    def rewind(self, parts: int) -> None:
        """Keep only the first ``parts`` parts, the ones a checkpoint counted (ADR 0009).

        Raises:
            ValueError: If any of them is missing, as after the run was joined.
        """
        self.require_parts(parts)
        kept = [self._part(index).name for index in range(parts)]
        for part in sorted(self._parts.glob("*.parquet")) if self._parts.is_dir() else []:
            if part.name not in kept:
                part.unlink()
        self.parts_written = parts

    def join(self) -> None:
        """Join the parts in the final file, in blocks of ``flush_rows`` rows (§11.1).

        The parts stay, so that an interrupted end can still resume from a checkpoint.
        """
        self.flush()
        temporary = self._path.with_name(self._path.name + ".tmp")
        with pq.ParquetWriter(temporary, self._schema) as writer:
            remainder = self._schema.empty_table()
            for index in range(self.parts_written):
                remainder = pa.concat_tables([remainder, pq.read_table(self._part(index))])
                remainder = self._write_full_blocks(writer, remainder)
            if remainder.num_rows:
                writer.write_table(remainder)
        temporary.replace(self._path)

    def discard_parts(self) -> None:
        """Delete the parts, once the end of the run is on record (ADR 0009)."""
        if self._parts.is_dir():
            shutil.rmtree(self._parts)

    def _write_full_blocks(self, writer: pq.ParquetWriter, table: pa.Table) -> pa.Table:
        start = 0
        while table.num_rows - start >= self._flush_rows:
            writer.write_table(table.slice(start, self._flush_rows))
            start += self._flush_rows
        return table.slice(start)

    def _part(self, index: int) -> Path:
        return self._parts / f"{index:0{PART_DIGITS}d}.parquet"


@dataclass(slots=True)
class Interval:
    """What the next row of ``ticks`` sums up (§11.5, ADR 0009).

    Attributes:
        ticks: Ticks since the last row.
        births, deaths: Their totals.
        worst_energy_error, worst_water_error: The largest errors of the balances (§7.6).
    """

    ticks: int = 0
    births: int = 0
    deaths: int = 0
    worst_energy_error: float = 0.0
    worst_water_error: float = 0.0

    def add(self, report: TickReport) -> None:
        """Count one tick."""
        self.ticks += 1
        self.births += report.births
        self.deaths += report.deaths
        self.worst_energy_error = max(self.worst_energy_error, report.balance.energy_error)
        self.worst_water_error = max(self.worst_water_error, report.balance.water_error)


class TelemetryRecorder:
    """Writes the tables of §11.5 for one world; the state is only read."""

    def __init__(self, output: Path, lab: LabConfig) -> None:
        """Prepare the writers of a run under ``output``."""
        self._t_log = lab.t_log
        self._ticks = BlockWriter(output / TICKS_FILE, TICKS_SCHEMA, lab.flush_rows)
        self._events = BlockWriter(output / EVENTS_FILE, EVENTS_SCHEMA, lab.flush_rows)
        self._interval = Interval()

    def start[G](self, simulation: Simulation[G]) -> None:
        """Record the founders as births of tick 0, the roots of the genealogy (ADR 0009)."""
        state = simulation.state
        self._events.add(_birth_events(state, state.pool.active(), tick=0, founders=True))

    def record[G](self, simulation: Simulation[G], report: TickReport) -> None:
        """Record one tick, after ``step_batch`` and before the next one (§2.4 step 9).

        The dead keep their data in their slots until the next tick (§7.5, ADR 0006).
        """
        state = simulation.state
        if report.births:
            self._events.add(_birth_events(state, report.born, report.tick, founders=False))
        if report.deaths:
            self._events.add(_death_events(state, report))
        self._interval.add(report)
        if (report.tick + 1) % self._t_log == 0:
            self._write_row(simulation, report.tick)

    def checkpoint_arrays(self) -> tuple[NamedArray, ...]:
        """Write everything pending and return what a checkpoint must keep (ADR 0009)."""
        self._ticks.flush()
        self._events.flush()
        i = self._interval
        return (
            ("ticks_parts", np.array([self._ticks.parts_written], dtype=np.int64)),
            ("events_parts", np.array([self._events.parts_written], dtype=np.int64)),
            ("interval", np.array([i.ticks, i.births, i.deaths], dtype=np.int64)),
            (
                "worst_errors",
                np.array([i.worst_energy_error, i.worst_water_error], dtype=np.float64),
            ),
        )

    def require_resumable(self, extras: dict[str, npt.NDArray[np.generic]]) -> None:
        """Check that the parts a checkpoint counted are all there, dropping nothing.

        Raises:
            ValueError: If a part is missing, as after the run was joined.
        """
        self._ticks.require_parts(int(extras["ticks_parts"][0]))
        self._events.require_parts(int(extras["events_parts"][0]))

    def resume(self, extras: dict[str, npt.NDArray[np.generic]]) -> None:
        """Return to what ``checkpoint_arrays`` returned, dropping the parts after it.

        Raises:
            ValueError: If a part is missing, as after the run was joined.
        """
        self.require_resumable(extras)
        self._ticks.rewind(int(extras["ticks_parts"][0]))
        self._events.rewind(int(extras["events_parts"][0]))
        ticks, births, deaths = (int(value) for value in extras["interval"])
        energy, water = (float(value) for value in extras["worst_errors"])
        self._interval = Interval(ticks, births, deaths, energy, water)

    def finish[G](self, simulation: Simulation[G]) -> None:
        """Write the row of an unfinished interval and the final files (§11.1, ADR 0009)."""
        if self._interval.ticks:
            self._write_row(simulation, simulation.t - 1)
        self._ticks.join()
        self._events.join()

    def discard_parts(self) -> None:
        """Delete the parts of both tables, after the manifest records the end (ADR 0009)."""
        self._ticks.discard_parts()
        self._events.discard_parts()

    def _write_row[G](self, simulation: Simulation[G], tick: int) -> None:
        self._ticks.add(tick_row(simulation, tick, self._interval))
        self._interval = Interval()


def tick_row[G](simulation: Simulation[G], tick: int, interval: Interval) -> pa.Table:
    """One row of ``ticks``: the state at the end of ``tick`` and the interval (§11.5)."""
    environment, state = simulation.environment, simulation.state
    plants, climate, water = environment.plants, environment.climate, environment.water
    biomass = plants.biomass[plants.pool.active()]
    stack = np.ceil(biomass)
    live = state.pool.active()
    row: dict[str, list[Any]] = {
        "tick": [tick],
        "population": [int(live.size)],
        "births": [interval.births],
        "deaths": [interval.deaths],
        "plants_total": [int(biomass.size)],
        "biomass_total": [float(np.sum(biomass))],
        "water_lakes": [float(np.sum(water.lakes.volume))],
        "water_puddles": [float(np.sum(water.puddles.volume))],
        "light": [climate.light(tick)],
        "rain": [climate.is_raining(tick)],
        "humidity": [climate.humidity],
        "mean_energy": [float(np.mean(state.energy[live])) if live.size else None],
        "mean_hydration": [float(np.mean(state.hydration[live])) if live.size else None],
        "species_count": [None],
        "energy_ledger_error": [interval.worst_energy_error],
        "water_ledger_error": [interval.worst_water_error],
    }
    for level in STACKS:
        row[f"plants_stack{level}"] = [int(np.count_nonzero(stack == level))]
    return pa.Table.from_pydict(row, schema=TICKS_SCHEMA)


def _birth_events(state: CreatureState, slots: IntArray, tick: int, *, founders: bool) -> pa.Table:
    labels = [None if founders else ASEXUAL] * slots.size
    return _events(state, slots, state.pool.ids[slots], tick, BIRTH, labels, [None] * slots.size)


def _death_events(state: CreatureState, report: TickReport) -> pa.Table:
    dead = report.dead
    labels = [None] * dead.count
    return _events(state, dead.slots, dead.uids, report.tick, DEATH, labels, dead.cause_labels())


def _events(
    state: CreatureState,
    slots: IntArray,
    uids: IntArray,
    tick: int,
    event: str,
    reproduction: Sequence[str | None],
    causes: Sequence[str | None],
) -> pa.Table:
    """The rows of one kind of event, one per slot, in ``uid`` order (§11.5)."""
    n = int(slots.size)
    parent = state.parent_uid[slots]
    absent = pa.nulls(n, pa.int64())
    columns = [
        pa.array(np.full(n, tick, dtype=np.int64)),
        pa.array([event] * n, pa.string()),
        pa.array(uids),
        pa.array(parent, mask=parent == NO_ID),
        absent,
        pa.array(state.founder_id[slots]),
        pa.array(reproduction, pa.string()),
        pa.array(causes, pa.string()),
        pa.array(state.age[slots]),
        pa.array(state.mass[slots]),
        pa.array(state.v_max[slots]),
        pa.array(state.color[slots]),
        absent,
    ]
    return pa.Table.from_arrays(columns, schema=EVENTS_SCHEMA)
