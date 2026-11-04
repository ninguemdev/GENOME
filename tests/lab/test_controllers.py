import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

from genome.core.config import GenomeConfig
from genome.core.protocols import (
    NUM_INPUTS,
    NUM_OUTPUTS,
    OUTPUT_DRINK,
    OUTPUT_EAT,
    OUTPUT_FORWARD,
    OUTPUT_LATERAL,
    OUTPUT_MEMORY,
    OUTPUT_REPRODUCE,
    OUTPUT_TURN,
    Brain,
    GenomeOperators,
)
from genome.core.rng import continuous_world_streams
from genome.core.sensors import observation as obs
from genome.lab.clonal import ClonalOperators
from genome.lab.controllers import (
    FixedGenome,
    HeuristicController,
    ImmobileController,
    RandomController,
)

ONE = np.array([0], dtype=np.int64)
FOOD = obs.INPUT_VISION_FOOD
WATER = obs.INPUT_VISION_WATER
DENSITY = obs.INPUT_VISION_DENSITY


def inputs(**values: float) -> np.ndarray:
    """One creature, comfortable and seeing nothing, with some inputs overridden."""
    row = np.zeros((1, NUM_INPUTS))
    row[0, obs.INPUT_ENERGY] = 0.5
    row[0, obs.INPUT_HYDRATION] = 0.6
    names = {
        "energy": obs.INPUT_ENERGY,
        "water": obs.INPUT_HYDRATION,
        "memory": obs.INPUT_MEMORY,
        "touch_food": obs.INPUT_TOUCH_FOOD,
        "touch_water": obs.INPUT_TOUCH_WATER,
    }
    for name, value in values.items():
        row[0, names[name]] = value
    return row


def decide(config: GenomeConfig, row: np.ndarray) -> np.ndarray:
    outputs: np.ndarray = HeuristicController(config).evaluate(row, ONE)[0]
    return outputs


def test_controllers_and_operators_implement_the_protocols(config: GenomeConfig) -> None:
    rng = continuous_world_streams(1).creatures
    for brain in (HeuristicController(config), ImmobileController(), RandomController(rng)):
        assert isinstance(brain, Brain)
        brain.assign(0, FixedGenome())
        brain.release(0)
    assert isinstance(ClonalOperators(), GenomeOperators)


def test_immobile_never_moves_and_always_tries(config: GenomeConfig) -> None:
    """§10.2: it never moves, and tries to eat and drink in every tick."""
    outputs = ImmobileController().evaluate(np.zeros((2, NUM_INPUTS)), np.arange(2, dtype=np.int64))
    assert outputs.shape == (2, NUM_OUTPUTS)
    assert np.all(outputs[:, [OUTPUT_FORWARD, OUTPUT_LATERAL, OUTPUT_TURN]] == 0.0)
    assert np.all(outputs[:, [OUTPUT_EAT, OUTPUT_DRINK]] > config.metabolism.eat_threshold)
    assert np.all(outputs[:, OUTPUT_REPRODUCE] == 0.0)


def test_random_draws_in_the_range_of_each_output() -> None:
    controller = RandomController(continuous_world_streams(1).creatures)
    outputs = controller.evaluate(np.zeros((500, NUM_INPUTS)), np.arange(500, dtype=np.int64))
    assert outputs[:, :3].min() < -0.9
    assert outputs[:, :3].max() <= 1.0
    assert outputs[:, 3:].min() >= 0.0
    assert outputs[:, 3:].max() > 0.9


def test_random_is_reproducible_from_its_stream() -> None:
    def draw() -> np.ndarray:
        controller = RandomController(continuous_world_streams(5).creatures)
        outputs: np.ndarray = controller.evaluate(np.zeros((3, NUM_INPUTS)), ONE.repeat(3))
        return outputs

    assert np.array_equal(draw(), draw())


def test_explores_walking_and_turning_when_nothing_is_seen(config: GenomeConfig) -> None:
    """A quarter of the phase: the turn is the whole amplitude (ADR 0006)."""
    outputs = decide(config, inputs(memory=0.25))
    assert_allclose(outputs[OUTPUT_FORWARD], config.lab.heuristic_forward)
    assert_allclose(outputs[OUTPUT_TURN], config.lab.heuristic_explore_turn)
    assert outputs[OUTPUT_EAT] == outputs[OUTPUT_DRINK] == 0.0


def test_memory_advances_the_exploration_phase_and_wraps(config: GenomeConfig) -> None:
    step = config.lab.heuristic_memory_step
    assert_allclose(decide(config, inputs(memory=0.5))[OUTPUT_MEMORY], 0.5 + step)
    assert_allclose(decide(config, inputs(memory=1.0 - step / 2))[OUTPUT_MEMORY], step / 2)


def test_steers_to_the_strongest_food(config: GenomeConfig) -> None:
    """§10.2: the receptor at +35° is the strongest, so 2 · 35° in radians, limited to 1."""
    row = inputs()
    row[0, FOOD + 4] = 0.4
    row[0, FOOD + 1] = 0.3  # weaker food at −35° does not pull the turn
    assert_allclose(decide(config, row)[OUTPUT_TURN], min(1.0, 2.0 * math.radians(35.0)))


def test_equally_strong_food_goes_to_the_first_receptor(config: GenomeConfig) -> None:
    """The tie has no draw and does not leave the creature between two plants."""
    row = inputs()
    row[0, FOOD + 2] = row[0, FOOD + 3] = 0.5
    assert_allclose(decide(config, row)[OUTPUT_TURN], 2.0 * math.radians(-10.0))


def test_turn_is_limited(config: GenomeConfig) -> None:
    """The receptor at −70° asks for −2.44 and receives −1."""
    row = inputs()
    row[0, FOOD] = 0.2
    assert decide(config, row)[OUTPUT_TURN] == -1.0


def test_thirst_with_water_in_sight_wins_over_food(config: GenomeConfig) -> None:
    row = inputs(water=0.49)
    row[0, FOOD + 4] = 0.9  # food to the right
    row[0, WATER + 1] = 0.1  # water to the left
    assert decide(config, row)[OUTPUT_TURN] < 0.0


def test_thirst_without_water_in_sight_still_follows_food(config: GenomeConfig) -> None:
    row = inputs(water=0.49)
    row[0, FOOD + 4] = 0.9
    assert decide(config, row)[OUTPUT_TURN] > 0.0


def test_water_in_sight_is_ignored_without_thirst(config: GenomeConfig) -> None:
    row = inputs(water=0.5)
    row[0, FOOD + 4] = 0.9
    row[0, WATER + 1] = 0.9
    assert decide(config, row)[OUTPUT_TURN] > 0.0


@pytest.mark.parametrize(
    ("water", "drinks"),
    [(0.1, True), (0.89, True), (0.9, False)],
)
def test_touching_water_it_drinks_until_full_enough(
    config: GenomeConfig, water: float, drinks: bool
) -> None:
    """Otherwise it could never reach the hydration that reproducing asks for (ADR 0007)."""
    outputs = decide(config, inputs(water=water, touch_water=1.0))
    assert (outputs[OUTPUT_DRINK] > config.metabolism.drink_threshold) == drinks
    assert (outputs[OUTPUT_FORWARD] == 0.0) == drinks


def test_drink_until_passes_the_hydration_needed_to_reproduce(config: GenomeConfig) -> None:
    lab = config.lab
    assert lab.heuristic_drink_until > lab.heuristic_reproduce_hydration
    assert lab.heuristic_reproduce_hydration >= config.metabolism.repr_min_hydration_fraction
    assert lab.heuristic_reproduce_energy >= config.metabolism.repr_min_energy_fraction


@pytest.mark.parametrize(
    ("energy", "eats"),
    [(0.89, True), (0.9, False)],
)
def test_eats_and_stops_on_a_plant_only_when_hungry(
    config: GenomeConfig, energy: float, eats: bool
) -> None:
    """§10.2: eat on touching a plant when E < 0.9."""
    outputs = decide(config, inputs(energy=energy, touch_food=1.0))
    assert (outputs[OUTPUT_EAT] > config.metabolism.eat_threshold) == eats
    assert (outputs[OUTPUT_FORWARD] == 0.0) == eats


def test_seeking_water_does_not_stop_on_a_plant(config: GenomeConfig) -> None:
    row = inputs(water=0.2, energy=0.2, touch_food=1.0)
    row[0, WATER + 3] = 0.5
    outputs = decide(config, row)
    assert outputs[OUTPUT_FORWARD] > 0.0
    assert outputs[OUTPUT_EAT] > config.metabolism.eat_threshold


@pytest.mark.parametrize(
    ("energy", "water", "density", "wants"),
    [
        (0.93, 0.7, 0.0, True),
        (0.93, 0.7, 0.02, True),
        (0.92, 0.7, 0.0, False),
        (0.93, 0.69, 0.0, False),
        (0.93, 0.7, 0.03, False),
    ],
)
def test_reproduces_only_with_room_and_reserves(
    config: GenomeConfig, energy: float, water: float, density: float, wants: bool
) -> None:
    """ADR 0007: with slack in the reserves, and with few others in sight."""
    row = inputs(energy=energy, water=water)
    row[0, DENSITY + 5] = density
    outputs = decide(config, row)
    assert (outputs[OUTPUT_REPRODUCE] > config.metabolism.reproduce_threshold) == wants


def test_clone_is_the_same_immutable_genome() -> None:
    genome = FixedGenome()
    assert ClonalOperators().clone(genome) is genome


def test_other_operators_arrive_in_f3() -> None:
    operators = ClonalOperators()
    rng = continuous_world_streams(1).creatures
    genome = FixedGenome()
    with pytest.raises(NotImplementedError, match="F3"):
        operators.mutate(genome, rng)
    with pytest.raises(NotImplementedError, match="F3"):
        operators.crossover(genome, genome, 0.0, 0.0, rng)
    with pytest.raises(NotImplementedError, match="F3"):
        operators.distance(genome, genome)
