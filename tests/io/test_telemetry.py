import dataclasses
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from numpy.testing import assert_array_equal

from genome.core.config import GenomeConfig
from genome.core.tick import Simulation
from genome.io.telemetry import (
    EVENTS_FILE,
    EVENTS_SCHEMA,
    TICKS_FILE,
    TICKS_SCHEMA,
    BlockWriter,
    Interval,
    TelemetryRecorder,
    tick_row,
)
from genome.lab.controllers import FixedGenome
from genome.lab.ecology import make_simulation

Sim = Simulation[FixedGenome]
SCHEMA = pa.schema([("x", pa.int64())])


def lively(config: GenomeConfig, ticks: int = 60, t_log: int = 10) -> GenomeConfig:
    """Adults and quick births, in a world small enough to compare by hand."""
    run = dataclasses.replace(config.run, max_ticks=ticks, max_population=60)
    world = dataclasses.replace(config.world, initial_population=40)
    training = dataclasses.replace(config.training, n=40, t_max=ticks)
    creatures = dataclasses.replace(
        config.creatures,
        founder_energy_fraction=1.0,
        founder_hydration_fraction=0.2,
        t_mat_fraction=0.001,
    )
    metabolism = dataclasses.replace(config.metabolism, t_rec=5, repr_min_hydration_fraction=0.1)
    lab = dataclasses.replace(config.lab, t_log=t_log, flush_rows=50)
    return dataclasses.replace(
        config,
        run=run,
        world=world,
        training=training,
        creatures=creatures,
        metabolism=metabolism,
        lab=lab,
    )


def recorded(config: GenomeConfig, output: Path) -> Sim:
    simulation = make_simulation(config, 3, "random")
    recorder = TelemetryRecorder(output, config.lab)
    recorder.start(simulation)
    while not simulation.finished:
        recorder.record(simulation, simulation.step_batch())
    recorder.finish(simulation)
    return simulation


def batch(*values: int) -> pa.Table:
    return pa.table({"x": list(values)}, schema=SCHEMA)


def test_tick_row_with_values_calculated_by_hand(config: GenomeConfig) -> None:
    simulation = make_simulation(lively(config), 1, "immobile")
    environment = simulation.environment
    plants, state = environment.plants, simulation.state
    live_plants = plants.pool.active()
    plants.remove(live_plants[5:])
    plants.biomass[live_plants[:5]] = [0.5, 1.0, 1.5, 2.0, 3.0]
    environment.water.lakes.volume[:] = 2.0
    environment.water.puddles.volume[:] = 0.5
    environment.climate.humidity = 0.25
    live = state.pool.active()
    state.remove(live[4:])
    state.energy[live[:4]] = [10.0, 20.0, 30.0, 40.0]
    state.hydration[live[:4]] = [1.0, 2.0, 3.0, 6.0]
    interval = Interval(10, 7, 2, 1e-12, 3e-12)

    row = tick_row(simulation, 0, interval).to_pylist()[0]

    assert row["tick"] == 0
    assert (row["population"], row["births"], row["deaths"]) == (4, 7, 2)
    assert row["plants_total"] == 5
    assert (row["plants_stack1"], row["plants_stack2"], row["plants_stack3"]) == (2, 2, 1)
    assert row["biomass_total"] == 8.0
    assert row["water_lakes"] == 2.0 * environment.water.lakes.count
    assert row["water_puddles"] == 0.5 * environment.water.puddles.count
    assert row["light"] == environment.climate.light(0)
    assert row["rain"] is False
    assert row["humidity"] == 0.25
    assert (row["mean_energy"], row["mean_hydration"]) == (25.0, 3.0)
    assert row["species_count"] is None
    assert (row["energy_ledger_error"], row["water_ledger_error"]) == (1e-12, 3e-12)


def test_an_empty_world_has_no_means(config: GenomeConfig) -> None:
    simulation = make_simulation(lively(config), 1, "immobile")
    simulation.state.remove(simulation.state.pool.active())
    row = tick_row(simulation, 0, Interval()).to_pylist()[0]
    assert (row["mean_energy"], row["mean_hydration"]) == (None, None)


def test_schemas_hold_the_columns_of_the_specification() -> None:
    assert TICKS_SCHEMA.names == [
        "tick",
        "population",
        "births",
        "deaths",
        "plants_total",
        "plants_stack1",
        "plants_stack2",
        "plants_stack3",
        "biomass_total",
        "water_lakes",
        "water_puddles",
        "light",
        "rain",
        "humidity",
        "mean_energy",
        "mean_hydration",
        "species_count",
        "energy_ledger_error",
        "water_ledger_error",
    ]
    assert EVENTS_SCHEMA.names == [
        "tick",
        "event",
        "uid",
        "parent_a",
        "parent_b",
        "founder_id",
        "reproduction",
        "death_cause",
        "age",
        "mass",
        "v_max",
        "color",
        "species_id",
    ]


def test_rows_sum_up_their_intervals_and_the_last_one_is_written(
    config: GenomeConfig, tmp_path: Path
) -> None:
    chosen = lively(config, ticks=25, t_log=10)
    simulation = make_simulation(chosen, 3, "random")
    recorder = TelemetryRecorder(tmp_path, chosen.lab)
    recorder.start(simulation)
    births, deaths, errors = [], [], []
    while not simulation.finished:
        report = simulation.step_batch()
        births.append(report.births)
        deaths.append(report.deaths)
        errors.append(report.balance.energy_error)
        recorder.record(simulation, report)
    recorder.finish(simulation)
    ticks = pq.read_table(tmp_path / TICKS_FILE).to_pydict()
    assert ticks["tick"] == [9, 19, 24]
    assert ticks["births"] == [sum(births[:10]), sum(births[10:20]), sum(births[20:])]
    assert ticks["deaths"] == [sum(deaths[:10]), sum(deaths[10:20]), sum(deaths[20:])]
    assert ticks["energy_ledger_error"] == [max(errors[:10]), max(errors[10:20]), max(errors[20:])]


def test_an_interval_that_ends_with_the_run_gets_no_empty_row(
    config: GenomeConfig, tmp_path: Path
) -> None:
    recorded(lively(config, ticks=20, t_log=10), tmp_path)
    assert pq.read_table(tmp_path / TICKS_FILE).column("tick").to_pylist() == [9, 19]


def test_events_rebuild_the_genealogy(config: GenomeConfig, tmp_path: Path) -> None:
    """The final population is what the events say: births minus deaths (§11.5)."""
    chosen = lively(config, ticks=80)
    simulation = recorded(chosen, tmp_path)
    events = pq.read_table(tmp_path / EVENTS_FILE).to_pylist()
    births = [row for row in events if row["event"] == "birth"]
    deaths = [row for row in events if row["event"] == "death"]
    born = {row["uid"]: row for row in births}
    assert len(born) == len(births) == simulation.state.pool.next_id
    alive = set(born) - {row["uid"] for row in deaths}
    assert alive == set(simulation.state.pool.ids[simulation.state.pool.active()].tolist())
    founders = [row for row in births if row["parent_a"] is None]
    assert len(founders) == 40
    assert all(row["tick"] == 0 and row["reproduction"] is None for row in founders)
    children = [row for row in births if row["parent_a"] is not None]
    assert children
    for child in children:
        assert child["reproduction"] == "asexual"
        assert born[child["parent_a"]]["founder_id"] == child["founder_id"]
        assert child["age"] == 0


def test_death_events_carry_cause_age_and_body(config: GenomeConfig, tmp_path: Path) -> None:
    chosen = lively(config, ticks=80)
    simulation = make_simulation(chosen, 3, "random")
    recorder = TelemetryRecorder(tmp_path, chosen.lab)
    recorder.start(simulation)
    expected = {}
    while not simulation.finished:
        ages = simulation.state.age.copy()
        report = simulation.step_batch()
        for uid, slot, cause in zip(
            report.dead.uids.tolist(),
            report.dead.slots.tolist(),
            report.dead.cause_labels(),
            strict=True,
        ):
            expected[uid] = (report.tick, int(ages[slot]), cause)
        recorder.record(simulation, report)
    recorder.finish(simulation)
    events = pq.read_table(tmp_path / EVENTS_FILE).to_pylist()
    deaths = [row for row in events if row["event"] == "death"]
    born = {row["uid"]: row for row in events if row["event"] == "birth"}
    assert expected
    assert {row["uid"]: (row["tick"], row["age"], row["death_cause"]) for row in deaths} == expected
    for row in deaths:
        assert row["reproduction"] is None
        body = born[row["uid"]]
        assert (row["mass"], row["v_max"], row["color"]) == (
            body["mass"],
            body["v_max"],
            body["color"],
        )
        assert (row["parent_a"], row["founder_id"]) == (body["parent_a"], body["founder_id"])


def test_block_writer_writes_blocks_of_flush_rows(tmp_path: Path) -> None:
    writer = BlockWriter(tmp_path / "t.parquet", SCHEMA, flush_rows=4)
    for start in range(0, 10, 2):
        writer.add(batch(start, start + 1))
        assert writer.parts_written == (start + 2) // 4
    assert sorted(p.name for p in (tmp_path / "t.parquet.parts").iterdir()) == [
        "000000.parquet",
        "000001.parquet",
    ]
    writer.finish()
    assert not (tmp_path / "t.parquet.parts").exists()
    stored = pq.ParquetFile(tmp_path / "t.parquet")
    groups = [stored.metadata.row_group(i).num_rows for i in range(stored.num_row_groups)]
    assert groups == [4, 4, 2]
    assert stored.read().column("x").to_pylist() == list(range(10))


def test_forced_flushes_do_not_change_the_final_blocks(tmp_path: Path) -> None:
    """A checkpoint writes small parts; the joined file must not show them (ADR 0009)."""
    writer = BlockWriter(tmp_path / "t.parquet", SCHEMA, flush_rows=4)
    for value in range(10):
        writer.add(batch(value))
        if value in (0, 4, 5):
            writer.flush()
    writer.finish()
    stored = pq.ParquetFile(tmp_path / "t.parquet")
    groups = [stored.metadata.row_group(i).num_rows for i in range(stored.num_row_groups)]
    assert groups == [4, 4, 2]
    assert stored.read().column("x").to_pylist() == list(range(10))


def test_a_table_without_rows_is_still_a_valid_file(tmp_path: Path) -> None:
    BlockWriter(tmp_path / "t.parquet", SCHEMA, flush_rows=4).finish()
    stored = pq.read_table(tmp_path / "t.parquet")
    assert stored.num_rows == 0
    assert stored.schema.equals(SCHEMA)


def test_rewind_drops_later_parts_and_refuses_missing_ones(tmp_path: Path) -> None:
    writer = BlockWriter(tmp_path / "t.parquet", SCHEMA, flush_rows=1)
    for value in range(3):
        writer.add(batch(value))
    again = BlockWriter(tmp_path / "t.parquet", SCHEMA, flush_rows=1)
    again.rewind(1)
    again.add(batch(9))
    again.finish()
    assert pq.read_table(tmp_path / "t.parquet").column("x").to_pylist() == [0, 9]
    with pytest.raises(ValueError, match="expected 2 parts"):
        BlockWriter(tmp_path / "t.parquet", SCHEMA, flush_rows=1).rewind(2)


def test_recorder_state_survives_a_checkpoint(config: GenomeConfig, tmp_path: Path) -> None:
    """Part of an interval before the checkpoint and part after it give the same row."""
    chosen = lively(config, ticks=25)
    direct_dir, split_dir = tmp_path / "direct", tmp_path / "split"
    direct_dir.mkdir()
    split_dir.mkdir()
    recorded(chosen, direct_dir)

    simulation = make_simulation(chosen, 3, "random")
    recorder = TelemetryRecorder(split_dir, chosen.lab)
    recorder.start(simulation)
    for _ in range(14):
        recorder.record(simulation, simulation.step_batch())
    arrays = dict(recorder.checkpoint_arrays())
    continued = TelemetryRecorder(split_dir, chosen.lab)
    continued.resume(arrays)
    while not simulation.finished:
        continued.record(simulation, simulation.step_batch())
    continued.finish(simulation)

    for name in (TICKS_FILE, EVENTS_FILE):
        assert pq.read_table(split_dir / name).equals(pq.read_table(direct_dir / name))
    assert_array_equal(arrays["interval"][:1], np.array([4]))
