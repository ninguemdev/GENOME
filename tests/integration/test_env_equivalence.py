"""The same seed and the same actions give identical states on both paths (ARQUITETURA §2.2)."""

import dataclasses

import numpy as np
import pytest
from tests.conftest import CONFIGS

from genome.core.config import GenomeConfig, load_config
from genome.core.protocols import NUM_OUTPUTS
from genome.core.state_hash import state_hash
from genome.core.tick import TickReport
from genome.lab.ecology import make_simulation
from genome.lab.environments import fixed_controller, make_env

ACTION_SEED = 77
COMPARE_EVERY = 20


def scripted_actions(tick: int, capacity: int) -> np.ndarray:
    """Actions of one tick per slot, in ``float32`` as the action space delivers them."""
    rng = np.random.default_rng((ACTION_SEED, tick))
    actions = rng.uniform(0.0, 1.0, (capacity, NUM_OUTPUTS)).astype(np.float32)
    actions[:, :3] = rng.uniform(-1.0, 1.0, (capacity, 3)).astype(np.float32)
    return actions


class ScriptedBrain:
    """The fast path receiving the same actions that the environment receives by ``step``."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.tick = 0

    def assign(self, slot: int, genome: object) -> None:
        pass

    def release(self, slot: int) -> None:
        pass

    def evaluate(self, observations: np.ndarray, slots: np.ndarray) -> np.ndarray:
        actions: np.ndarray = scripted_actions(self.tick, self.capacity)[slots].astype(np.float64)
        self.tick += 1
        return actions


def lively(config: GenomeConfig, ticks: int) -> GenomeConfig:
    """Adults and quick births, in a short run with births, deaths and reused slots."""
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
    lab = dataclasses.replace(config.lab, ledger_debug=True)
    return dataclasses.replace(
        config, run=run, world=world, training=training, creatures=creatures,
        metabolism=metabolism, lab=lab,
    )  # fmt: skip


def test_scripted_actions_give_identical_states(config: GenomeConfig) -> None:
    chosen = lively(config, 120)
    capacity = chosen.run.max_population
    fast = make_simulation(chosen, 3, "immobile")
    brain = ScriptedBrain(capacity)
    fast._brain = brain
    env = make_env(chosen)
    env.reset(seed=3)

    births = deaths = 0
    reused = False
    freed: set[int] = set()
    report: TickReport
    while env.agents:
        tick = fast.t
        rows = scripted_actions(tick, capacity)
        actions = {name: rows[env._slot_of[name]] for name in env.agents}
        report = fast.step_batch()
        env.step(actions)
        births += report.births
        deaths += report.deaths
        reused = reused or bool(freed & set(report.born.tolist()))
        freed |= set(report.dead.slots.tolist())
        if tick % COMPARE_EVERY == 0:
            assert state_hash(fast.named_arrays()) == state_hash(list(env.state().items()))
    assert state_hash(fast.named_arrays()) == state_hash(list(env.state().items()))
    assert births > 0
    assert deaths > 0
    assert reused


@pytest.mark.parametrize("controller", ["heuristic", "random", "immobile"])
def test_controllers_give_identical_states(config: GenomeConfig, controller: str) -> None:
    """A controller in the environment draws the same numbers as on the fast path (§2.5)."""
    chosen = lively(config, 60)
    fast = make_simulation(chosen, 5, controller)  # type: ignore[arg-type]
    env = make_env(chosen)
    env.reset(seed=5, options={"controller": fixed_controller(controller)})  # type: ignore[arg-type]
    while env.agents:
        fast.step_batch()
        env.step({})
    assert state_hash(fast.named_arrays()) == state_hash(list(env.state().items()))


@pytest.mark.slow
def test_heuristic_generation_gives_identical_states() -> None:
    """A whole generation of the preset of F1, with the births and deaths it really has."""
    preset = load_config(CONFIGS / "f1_ecology.yaml")
    run = dataclasses.replace(preset.run, max_ticks=2000)
    training = dataclasses.replace(preset.training, t_max=2000)
    chosen = dataclasses.replace(preset, run=run, training=training)
    fast = make_simulation(chosen, 1, "heuristic")
    env = make_env(chosen)
    env.reset(seed=1, options={"controller": fixed_controller("heuristic")})
    while env.agents:
        fast.step_batch()
        env.step({})
    assert fast.t == 2000
    assert state_hash(fast.named_arrays()) == state_hash(list(env.state().items()))
