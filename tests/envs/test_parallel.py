import dataclasses

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pettingzoo.test import parallel_api_test

from genome.core.config import GenomeConfig, load_config
from genome.core.protocols import NUM_OUTPUTS, OUTPUT_REPRODUCE
from genome.core.state_hash import state_hash
from genome.envs.parallel import GenomeParallelEnv
from genome.envs.spaces import action_box, observation_box
from genome.lab.controllers import FixedGenome
from genome.lab.environments import fixed_controller, make_env
from tests.conftest import CONFIGS

# A world ends with its agents gone, while most slots were never occupied (§2.2).
pytestmark = pytest.mark.filterwarnings("ignore:No agents present")


def sized(config: GenomeConfig, n: int = 8, capacity: int = 24, ticks: int = 40) -> GenomeConfig:
    """A small world, so that the adapter is what the test measures."""
    run = dataclasses.replace(config.run, max_population=capacity, max_ticks=ticks)
    world = dataclasses.replace(config.world, initial_population=n)
    training = dataclasses.replace(config.training, n=n, t_max=ticks)
    return dataclasses.replace(config, run=run, world=world, training=training)


def fertile(config: GenomeConfig, n: int = 8, capacity: int = 200) -> GenomeConfig:
    """Founders start full and are adults from their second tick, so births come right away."""
    creatures = dataclasses.replace(
        config.creatures,
        founder_energy_fraction=1.0,
        founder_hydration_fraction=1.0,
        t_mat_fraction=1e-6,
    )
    metabolism = dataclasses.replace(config.metabolism, t_rec=1)
    small = sized(config, n, capacity)
    return dataclasses.replace(small, creatures=creatures, metabolism=metabolism)


def parched(config: GenomeConfig, n: int = 8) -> GenomeConfig:
    """Founders start almost dry, so everybody dies of thirst within the run."""
    creatures = dataclasses.replace(config.creatures, founder_hydration_fraction=0.005)
    return dataclasses.replace(sized(config, n), creatures=creatures)


def rest(env: GenomeParallelEnv[FixedGenome]) -> dict[str, np.ndarray]:
    return {name: np.zeros(NUM_OUTPUTS, dtype=np.float32) for name in env.agents}


def wanting_children(env: GenomeParallelEnv[FixedGenome]) -> dict[str, np.ndarray]:
    actions = rest(env)
    for action in actions.values():
        action[OUTPUT_REPRODUCE] = 1.0
    return actions


def api_test(config: GenomeConfig) -> GenomeParallelEnv[FixedGenome]:
    env = make_env(config)
    env.action_space("creature_0000").seed(1)
    parallel_api_test(env, num_cycles=50)
    return env


def test_official_api_test_until_truncation(config: GenomeConfig) -> None:
    api_test(sized(config))


def test_official_api_test_until_extinction(config: GenomeConfig) -> None:
    api_test(parched(config))


def test_official_api_test_with_births_in_new_slots(config: GenomeConfig) -> None:
    """Without reuse: the official test forbids it, and §2.2 asks for it (ADR 0008)."""
    env = api_test(fertile(config))
    assert env._simulation is not None
    assert env._simulation.state.pool.next_id > 8


def test_metadata_and_agents(config: GenomeConfig) -> None:
    chosen = sized(config, capacity=12)
    env = make_env(chosen)
    assert env.metadata == {"name": "genome_v1", "render_modes": ["human", "rgb_array"]}
    assert env.possible_agents[0] == "creature_0000"
    assert env.possible_agents[-1] == "creature_0011"
    assert len(env.possible_agents) == 12


def test_spaces_match_the_specification() -> None:
    observation, action = observation_box(), action_box()
    assert observation.shape == (42,)
    assert observation.dtype == np.float32
    assert_array_equal(np.flatnonzero(observation.low < 0.0), [25, 27, 33, 34, 35])
    assert_array_equal(action.low, [-1, -1, -1, 0, 0, 0, 0])
    assert_array_equal(action.high, np.ones(7))


def test_reset_gives_every_founder_an_observation_and_an_identity(config: GenomeConfig) -> None:
    env = make_env(sized(config))
    observations, infos = env.reset(seed=1)
    assert env.agents == env.possible_agents[:8]
    assert set(observations) == set(infos) == set(env.agents)
    assert infos["creature_0003"] == {"uid": 3, "founder_id": 3, "born": 0, "death_cause": None}
    assert observations["creature_0003"] in env.observation_space("creature_0003")


def test_the_first_reset_needs_a_seed_and_later_ones_repeat_it(config: GenomeConfig) -> None:
    env = make_env(sized(config))
    with pytest.raises(ValueError, match="seed"):
        env.reset()
    first, _ = env.reset(seed=5)
    again, _ = env.reset()
    assert_array_equal(first["creature_0000"], again["creature_0000"])


def test_step_needs_a_reset_and_a_running_world(config: GenomeConfig) -> None:
    env = make_env(sized(config, ticks=1))
    with pytest.raises(RuntimeError, match="reset"):
        env.step({})
    env.reset(seed=1)
    env.step(rest(env))
    assert env.agents == []
    with pytest.raises(RuntimeError, match="reset"):
        env.step({})


def test_actions_must_match_the_live_agents(config: GenomeConfig) -> None:
    env = make_env(sized(config, n=2))
    env.reset(seed=1)
    actions = rest(env)
    with pytest.raises(ValueError, match="no action"):
        env.step({"creature_0000": actions["creature_0000"]})
    with pytest.raises(ValueError, match="not alive"):
        env.step({**actions, "creature_0005": np.zeros(NUM_OUTPUTS, dtype=np.float32)})


def test_actions_outside_the_ranges_are_limited(config: GenomeConfig) -> None:
    """Each value becomes its nearest bound before entering the core (§6.2)."""
    wild = make_env(sized(config, n=1))
    wild.reset(seed=1)
    wild.step({"creature_0000": np.array([2.0, -2.0, 2.0, -2.0, 2.0, -2.0, 2.0], dtype=np.float32)})
    limited = make_env(sized(config, n=1))
    limited.reset(seed=1)
    limited.step({"creature_0000": np.array([1, -1, 1, 0, 1, 0, 1], dtype=np.float32)})
    assert wild._simulation is not None
    assert wild._simulation.state.memory[0] == 1.0
    assert state_hash(list(wild.state().items())) == state_hash(list(limited.state().items()))


def test_nan_actions_are_errors(config: GenomeConfig) -> None:
    env = make_env(sized(config, n=1))
    env.reset(seed=1)
    with pytest.raises(ValueError, match="NaN"):
        env.step({"creature_0000": np.full(NUM_OUTPUTS, np.nan, dtype=np.float32)})


def test_action_of_the_wrong_size_is_an_error(config: GenomeConfig) -> None:
    env = make_env(sized(config, n=1))
    env.reset(seed=1)
    with pytest.raises(ValueError, match="7 values"):
        env.step({"creature_0000": np.zeros(3, dtype=np.float32)})


def test_death_is_reported_once_with_its_cause(config: GenomeConfig) -> None:
    """Hydration 0.3 of 60 ends in the first ticks; simultaneous causes join by + (ADR 0008)."""
    env = make_env(parched(config, n=1))
    env.reset(seed=1)
    deaths = []
    while env.agents:
        final = env.agents[0]
        _, rewards, terminations, truncations, infos = env.step(rest(env))
        deaths.append(terminations[final])
        assert not truncations[final]
    assert deaths[-1]
    assert sum(deaths) == 1
    assert infos[final]["death_cause"] == "dehydration"
    assert infos[final]["uid"] == 0
    assert rewards[final] == 0.0


def test_truncation_reaches_everyone_at_max_ticks(config: GenomeConfig) -> None:
    env = make_env(sized(config, n=3, ticks=2))
    env.reset(seed=1)
    _, _, _, truncations, _ = env.step(rest(env))
    assert not any(truncations.values())
    observations, _, terminations, truncations, _ = env.step(rest(env))
    assert all(truncations.values())
    assert not any(terminations.values())
    assert len(truncations) == 3
    assert len(observations) == 3
    assert env.agents == []


def test_newborn_appears_in_the_tick_of_its_birth(config: GenomeConfig) -> None:
    """It enters ``agents`` with its first observation in the return of the step (§2.2)."""
    env = make_env(fertile(config, n=1))
    env.reset(seed=1)
    env.step(rest(env))
    observations, rewards, terminations, _, infos = env.step(wanting_children(env))
    assert env.agents == ["creature_0000", "creature_0001"]
    newborn = "creature_0001"
    assert observations[newborn] in env.observation_space(newborn)
    assert infos[newborn] == {"uid": 1, "founder_id": 0, "born": 1, "death_cause": None}
    assert not terminations[newborn]
    assert rewards[newborn] == 0.0


def test_slot_returns_with_a_new_uid_only_after_the_death_was_reported(
    config: GenomeConfig,
) -> None:
    """§2.2: the name is the slot and the uid the identity. The official test forbids this."""
    env = make_env(fertile(config, n=2, capacity=2))
    env.reset(seed=1)
    simulation = env._simulation
    assert simulation is not None
    simulation.state.hydration[1] = 0.0
    _, _, terminations, _, infos = env.step(rest(env))
    assert terminations["creature_0001"]
    assert infos["creature_0001"]["uid"] == 1
    assert env.agents == ["creature_0000"]

    _, _, terminations, _, infos = env.step(wanting_children(env))
    assert env.agents == ["creature_0000", "creature_0001"]
    assert not terminations["creature_0001"]
    assert infos["creature_0001"]["uid"] == 2
    assert infos["creature_0001"]["founder_id"] == 0


def test_a_controller_decides_for_whoever_has_no_action(config: GenomeConfig) -> None:
    env = make_env(sized(config, n=2))
    env.reset(seed=1, options={"controller": fixed_controller("immobile")})
    env.step({})
    env.step({"creature_0000": np.zeros(NUM_OUTPUTS, dtype=np.float32)})
    assert env.agents == ["creature_0000", "creature_0001"]


def test_rewards_add_up_to_the_fitness_of_a_life(config: GenomeConfig) -> None:
    """The sum over the ticks lived is F_base (§2.2, §9.2), computed here from the reserves."""
    chosen = sized(config, n=1, ticks=30)
    env = make_env(chosen)
    env.reset(seed=1)
    simulation = env._simulation
    assert simulation is not None
    state, training = simulation.state, chosen.training
    total = expected = 0.0
    while env.agents:
        _, rewards, _, _, _ = env.step(rest(env))
        total += rewards["creature_0000"]
        wellbeing = min(state.energy[0] / state.e_max[0], state.hydration[0] / state.h_max[0])
        expected += (training.w_s + training.w_b * wellbeing) / training.t_max
    assert_allclose(total, expected)
    assert total > training.w_s  # it lived the whole t_max, so S = 1


def test_reward_counts_the_child_when_it_is_born(config: GenomeConfig) -> None:
    chosen = fertile(config, n=1)
    env = make_env(chosen)
    env.reset(seed=1)
    simulation = env._simulation
    assert simulation is not None
    state, training = simulation.state, chosen.training
    env.step(rest(env))
    _, rewards, _, _, _ = env.step(wanting_children(env))
    assert simulation.state.n_children[0] == 1
    wellbeing = min(state.energy[0] / state.e_max[0], state.hydration[0] / state.h_max[0])
    base = (training.w_s + training.w_b * wellbeing) / training.t_max
    bonus = training.k_repr * (1.0 - np.exp(-1.0 / training.d_sat))
    assert_allclose(rewards["creature_0000"], base + bonus)


def test_splits_choose_different_worlds(config: GenomeConfig) -> None:
    env = make_env(sized(config))
    hashes = set()
    for options in (
        {"split": "train"},
        {"split": "train", "generation": 1},
        {"split": "train", "world_index": 1},
        {"split": "val"},
        {"split": "test"},
    ):
        env.reset(seed=1, options=options)
        hashes.add(state_hash(list(env.state().items())))
    assert len(hashes) == 5
    with pytest.raises(ValueError, match="unknown split"):
        env.reset(seed=1, options={"split": "nope"})


def test_validation_and_test_worlds_ignore_the_root_seed(config: GenomeConfig) -> None:
    """They come from the laboratory seed (§2.5, §10.3)."""
    env = make_env(sized(config))
    env.reset(seed=1, options={"split": "val"})
    first = state_hash(list(env.state().items()))
    env.reset(seed=2, options={"split": "val"})
    assert state_hash(list(env.state().items())) == first


def test_continuous_mode_starts_in_f3(config: GenomeConfig) -> None:
    env = make_env(sized(config))
    with pytest.raises(NotImplementedError, match="F3"):
        env.reset(seed=1, options={"mode": "continuous"})
    preset = load_config(CONFIGS / "continuous.yaml")
    env = make_env(preset)
    with pytest.raises(NotImplementedError, match="F3"):
        env.reset(seed=1)


def test_state_is_a_copy_for_inspection(config: GenomeConfig) -> None:
    env = make_env(sized(config))
    with pytest.raises(RuntimeError, match="reset"):
        env.state()
    env.reset(seed=1)
    snapshot = env.state()
    snapshot["creatures.energy"][:] = -1.0
    assert env.state()["creatures.energy"][0] > 0.0


def test_render_is_delegated_to_the_injected_renderer(config: GenomeConfig) -> None:
    class Spy:
        def render(self, state: object) -> str:
            assert "creatures.position" in state  # type: ignore[operator]
            return "drawn"

    chosen = sized(config)
    env = GenomeParallelEnv(
        chosen, make_env(chosen)._operators, [FixedGenome()] * 8, "rgb_array", Spy()
    )
    env.reset(seed=1)
    assert env.render() == "drawn"


def test_render_without_a_renderer_is_an_error(config: GenomeConfig) -> None:
    env = make_env(sized(config))
    env.reset(seed=1)
    with pytest.raises(RuntimeError, match="render"):
        env.render()
    with pytest.raises(ValueError, match="render mode"):
        GenomeParallelEnv(sized(config), make_env(config)._operators, [], "ascii")
