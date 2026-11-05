"""Milliseconds per tick of each system, in reference worlds (PADROES §6, ARQUITETURA §12.6).

Up to 600 creatures, the world is the training one (256 × 256). Above that, it is the
geography of the continuous preset (512 × 512), at the density of §12.4, but still in
``mode: training``, because the continuous mode only exists from F3 on (ADR 0010).

The total is the median of runs without the profiler; the systems come from one more run
under it, which slows them down evenly, so read them as shares of the total. A few ticks
run first, outside the clock, so that compiling or loading the Numba kernels does not count.

With ``--processes``, it measures instead how the throughput scales with independent worlds,
one process each, all starting their clock together after loading the kernels.

Usage: ``uv run python scripts/bench.py --population 200 600 1000 --ticks 1000``
Scaling: ``uv run python scripts/bench.py --population 600 --processes 1 4 8 12 16``
"""

import argparse
import cProfile
import dataclasses
import multiprocessing
import multiprocessing.queues
import multiprocessing.synchronize
import os
import pstats
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import get_args

from genome.core.config import GenomeConfig, load_config
from genome.core.config.sections import ProcessPriority
from genome.core.tick import Simulation
from genome.io.telemetry import TelemetryRecorder
from genome.lab.controllers import FixedGenome
from genome.lab.ecology import make_simulation
from genome.lab.pool import SINGLE_THREAD_VARIABLES, set_process_priority

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
SEED = 1
WARMUP_TICKS = 5
#: Above this population the world is the one of the continuous preset (§12.1, §12.4).
TRAINING_WORLD_LIMIT = 600
FRAMES = 30
MILLISECONDS = 1000.0
# Systems reported, as (label, function name).
SYSTEMS = (
    ("sensors", "build_observations"),
    ("  vision", "vision_channels"),
    ("  pairs", "build_scene"),
    ("  smell", "olfaction_channels"),
    ("  touch", "touch_channels"),
    ("locomotion", "integrate_locomotion"),
    ("eating", "resolve_eating"),
    ("drinking", "resolve_drinking"),
    ("metabolism", "apply_metabolism"),
    ("environment", "step_environment"),
)
#: One thread per process: each library would otherwise start its own pool in every world.
BYTES_PER_MEBIBYTE = 2**20
KIBIBYTES_PER_MEBIBYTE = 2**10


@dataclasses.dataclass(frozen=True, slots=True)
class ScaleSample:
    """What one world measured while the others ran.

    Attributes:
        ms_per_tick: Milliseconds per tick of this world.
        user, kernel: Processor seconds in the program and in the operating system.
        wall: Seconds on the clock.
        page_faults: Page faults per tick.
        peak_mebibytes: Peak memory of the process.
    """

    ms_per_tick: float
    user: float
    kernel: float
    wall: float
    page_faults: float
    peak_mebibytes: float


if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes

    class _MemoryCounters(ctypes.Structure):
        """``PROCESS_MEMORY_COUNTERS`` of the Windows API."""

        _fields_ = (
            ("cb", wintypes.DWORD),
            ("page_fault_count", wintypes.DWORD),
            ("peak_working_set_size", ctypes.c_size_t),
            ("working_set_size", ctypes.c_size_t),
            ("quota_peak_paged_pool_usage", ctypes.c_size_t),
            ("quota_paged_pool_usage", ctypes.c_size_t),
            ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
            ("quota_non_paged_pool_usage", ctypes.c_size_t),
            ("pagefile_usage", ctypes.c_size_t),
            ("peak_pagefile_usage", ctypes.c_size_t),
        )

    def memory_counters() -> tuple[int, float]:
        """Page faults so far and peak memory of this process, in mebibytes."""
        counters = _MemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        current_process = ctypes.c_void_p(-1)
        ctypes.windll.psapi.GetProcessMemoryInfo(
            current_process, ctypes.byref(counters), counters.cb
        )
        return counters.page_fault_count, counters.peak_working_set_size / BYTES_PER_MEBIBYTE

else:
    import resource

    def memory_counters() -> tuple[int, float]:
        """Page faults so far and peak memory of this process, in mebibytes."""
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return usage.ru_minflt + usage.ru_majflt, usage.ru_maxrss / KIBIBYTES_PER_MEBIBYTE


def sized_config(population: int) -> GenomeConfig:
    """The reference world of ``population`` founders, in training mode (ADR 0010)."""
    base = load_config(CONFIGS / "f1_ecology.yaml")
    limit = base.training.t_max
    if population > TRAINING_WORLD_LIMIT:
        continuous = load_config(CONFIGS / "continuous.yaml")
        base = dataclasses.replace(
            continuous,
            run=dataclasses.replace(continuous.run, mode="training", max_ticks=limit),
            training=dataclasses.replace(continuous.training, t_max=limit),
        )
    run = dataclasses.replace(base.run, max_population=max(population, base.run.max_population))
    world = dataclasses.replace(base.world, initial_population=population)
    training = dataclasses.replace(base.training, n=population)
    return dataclasses.replace(base, run=run, world=world, training=training)


def warm_world(config: GenomeConfig) -> Simulation[FixedGenome]:
    """A world of ``config`` under the heuristic, after the warm-up ticks."""
    simulation = make_simulation(config, SEED, "heuristic")
    for _ in range(WARMUP_TICKS):
        simulation.step_batch()
    return simulation


def timed_run(config: GenomeConfig, ticks: int) -> tuple[float, int]:
    """Milliseconds per tick of one run, and the population at its end."""
    simulation = warm_world(config)
    start = time.perf_counter()
    for _ in range(ticks):
        simulation.step_batch()
    return MILLISECONDS * (time.perf_counter() - start) / ticks, simulation.state.pool.count


def recorded_run(config: GenomeConfig, ticks: int) -> float:
    """Milliseconds per tick of one run with the telemetry of §11.5, joining included."""
    simulation = warm_world(config)
    with tempfile.TemporaryDirectory() as directory:
        recorder = TelemetryRecorder(Path(directory), config.lab)
        recorder.start(simulation)
        start = time.perf_counter()
        for _ in range(ticks):
            recorder.record(simulation, simulation.step_batch())
        recorder.finish(simulation)
        elapsed = time.perf_counter() - start
    return MILLISECONDS * elapsed / ticks


def frame_cost(config: GenomeConfig) -> float:
    """Milliseconds to copy the arrays of the world and draw one frame, without a screen."""
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    # Imported here: the benchmark of the simulation does not need the extra ``viz``.
    from genome.viz.renderer import WorldRenderer

    simulation = warm_world(config)
    renderer = WorldRenderer(config)
    start = time.perf_counter()
    for _ in range(FRAMES):
        renderer.render({name: array.copy() for name, array in simulation.named_arrays()})
    return MILLISECONDS * (time.perf_counter() - start) / FRAMES


def shares(config: GenomeConfig, ticks: int) -> dict[str, float]:
    """Share of the tick spent in each system, under the profiler."""
    simulation = warm_world(config)
    profile = cProfile.Profile()
    profile.enable()
    for _ in range(ticks):
        simulation.step_batch()
    profile.disable()
    stats = pstats.Stats(profile).stats  # type: ignore[attr-defined]
    whole = sum(v[3] for k, v in stats.items() if k[2] == "step_batch")
    return {
        label: sum(v[3] for k, v in stats.items() if k[2] == name) / whole
        for label, name in SYSTEMS
    }


def bench(population: int, ticks: int, repeats: int) -> None:
    """Print the total, the share of each system and the costs around the tick."""
    config = sized_config(population)
    runs = [timed_run(config, ticks) for _ in range(repeats)]
    total = statistics.median(run[0] for run in runs)
    recorded = statistics.median(recorded_run(config, ticks) for _ in range(repeats))
    world = f"{config.world.width:.0f} x {config.world.height:.0f}"
    print(
        f"\npopulation {population} in {world}: {total:.2f} ms/tick over {ticks} ticks, "
        f"ended at {runs[0][1]}; runs {[round(run[0], 2) for run in runs]}"
    )
    for label, share in shares(config, ticks).items():
        print(f"  {label:<12} {100.0 * share:5.1f}%  ~{total * share:6.2f} ms")
    print(f"  with telemetry  {recorded:.2f} ms/tick ({recorded - total:+.2f} ms)")
    print(f"  one frame of the viewer  {frame_cost(config):.2f} ms")


def scale_worker(
    population: int,
    ticks: int,
    barrier: multiprocessing.synchronize.Barrier,
    samples: "multiprocessing.queues.Queue[ScaleSample]",
    priority: ProcessPriority = "normal",
) -> None:
    """Run one world, starting the clock together with the others."""
    set_process_priority(priority)
    simulation = warm_world(sized_config(population))
    barrier.wait()
    faults_before, _ = memory_counters()
    times_before = os.times()
    start = time.perf_counter()
    for _ in range(ticks):
        simulation.step_batch()
    wall = time.perf_counter() - start
    times_after = os.times()
    faults_after, peak = memory_counters()
    samples.put(
        ScaleSample(
            ms_per_tick=MILLISECONDS * wall / ticks,
            user=times_after.user - times_before.user,
            kernel=times_after.system - times_before.system,
            wall=wall,
            page_faults=(faults_after - faults_before) / ticks,
            peak_mebibytes=peak,
        )
    )


def measure_scale(
    population: int, ticks: int, processes: int, priority: ProcessPriority = "normal"
) -> list[ScaleSample]:
    """The samples of ``processes`` independent worlds running at the same time."""
    for variable in SINGLE_THREAD_VARIABLES:
        os.environ[variable] = "1"
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(processes)
    samples: multiprocessing.queues.Queue[ScaleSample] = context.Queue()
    workers = [
        context.Process(target=scale_worker, args=(population, ticks, barrier, samples, priority))
        for _ in range(processes)
    ]
    for worker in workers:
        worker.start()
    measured = [samples.get() for _ in workers]
    for worker in workers:
        worker.join()
    return measured


def report_scale(
    population: int, ticks: int, counts: list[int], priority: ProcessPriority = "normal"
) -> None:
    """Print the throughput and where the processor went, for each number of processes."""
    print(
        f"\npopulation {population}, {ticks} ticks, {os.cpu_count()} logical processors, "
        f"priority {priority}"
    )
    print("procs  ms/tick  ticks/s  gain  cpu use  kernel  faults/tick  peak MiB")
    single = None
    for processes in counts:
        samples = measure_scale(population, ticks, processes, priority)
        throughput = sum(MILLISECONDS / sample.ms_per_tick for sample in samples)
        single = single if single is not None else throughput / processes
        busy = sum(sample.user + sample.kernel for sample in samples)
        wall = max(sample.wall for sample in samples)
        print(
            f"{processes:5d}"
            f"  {statistics.median(sample.ms_per_tick for sample in samples):7.2f}"
            f"  {throughput:7.0f}"
            f"  {throughput / single:4.1f}"
            f"  {busy / (wall * (os.cpu_count() or 1)):7.0%}"
            f"  {sum(sample.kernel for sample in samples) / busy:6.0%}"
            f"  {statistics.median(sample.page_faults for sample in samples):11.0f}"
            f"  {max(sample.peak_mebibytes for sample in samples):8.0f}",
            flush=True,
        )


def main() -> None:
    """Run the benchmark for each population."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", type=int, nargs="+", default=[200, 600, 1000])
    parser.add_argument("--ticks", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--processes", type=int, nargs="+", default=None)
    parser.add_argument("--priority", default="normal", choices=get_args(ProcessPriority))
    args = parser.parse_args()
    for population in args.population:
        if args.processes:
            report_scale(population, args.ticks, args.processes, args.priority)
        else:
            bench(population, args.ticks, args.repeats)


if __name__ == "__main__":
    main()
