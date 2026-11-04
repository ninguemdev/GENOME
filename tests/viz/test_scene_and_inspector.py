"""What the viewer shows, without any window (ARQUITETURA §2.1, §5.1, §5.2, §5.3)."""

import dataclasses
import math

import numpy as np
import pytest

from genome.core.config import GenomeConfig
from genome.core.tick import Simulation
from genome.lab.controllers import FixedGenome
from genome.lab.ecology import make_simulation
from genome.viz.inspector import (
    LOBE_SAMPLES,
    PICK_REACH,
    inspect_creature,
    panel_lines,
    pick_creature,
    retina_lobes,
)
from genome.viz.scene import Frame, WorldArrays, build_frame, hue_to_rgb


def small(config: GenomeConfig) -> GenomeConfig:
    run = dataclasses.replace(config.run, max_population=30, max_ticks=50)
    world = dataclasses.replace(config.world, initial_population=12)
    training = dataclasses.replace(config.training, n=12, t_max=50)
    return dataclasses.replace(config, run=run, world=world, training=training)


def world(config: GenomeConfig) -> tuple[Simulation[FixedGenome], WorldArrays, Frame]:
    simulation = make_simulation(small(config), 3, "immobile")
    arrays = {name: array.copy() for name, array in simulation.named_arrays()}
    return simulation, arrays, build_frame(arrays, simulation.config)


@pytest.mark.parametrize(
    ("hue", "expected"),
    [
        # With S = 0.85 and V = 1: the two other channels stay at 255 · 0.15 ≈ 38.
        (0.0, (255, 38, 38)),
        (1.0 / 3.0, (38, 255, 38)),
        (2.0 / 3.0, (38, 38, 255)),
        # Halfway between red and green, red and green are both at the top.
        (1.0 / 6.0, (255, 255, 38)),
    ],
)
def test_the_color_gene_is_an_hsv_hue(hue: float, expected: tuple[int, int, int]) -> None:
    assert tuple(hue_to_rgb(np.array([hue]))[0].tolist()) == expected


def test_the_frame_shows_the_live_world(config: GenomeConfig) -> None:
    simulation, _, frame = world(config)
    pool = simulation.state.pool
    assert frame.tick == 0
    assert frame.size == (config.world.width, config.world.height)
    assert frame.uids.tolist() == pool.ids[pool.active()].tolist()
    assert frame.colors.shape == (12, 3)
    assert frame.plant_stacks.size == simulation.environment.plants.pool.count
    assert np.all((frame.plant_stacks >= 1) & (frame.plant_stacks <= 3))
    lakes = simulation.environment.water.lakes
    assert frame.lake_radii == pytest.approx(lakes.radius())
    assert frame.puddle_radii.size == 0  # Puddles start empty (§3.4).


def test_a_wet_puddle_has_the_radius_of_the_core(config: GenomeConfig) -> None:
    simulation, arrays, _ = world(config)
    water = simulation.config.water
    volume = arrays["puddles.volume"].copy()
    volume[0] = water.v_p / 4.0
    frame = build_frame({**arrays, "puddles.volume": volume}, simulation.config)
    # r = r_p,max · √(1/4) (§3.4).
    assert frame.puddle_radii.tolist() == pytest.approx([water.r_p_max / 2.0])


def test_a_click_selects_across_the_border_of_the_torus(config: GenomeConfig) -> None:
    _, _, frame = world(config)
    width = frame.size[0]
    target = dataclasses.replace(
        frame,
        uids=np.array([7, 8], dtype=np.int64),
        positions=np.array([[0.5, 100.0], [width / 2.0, 100.0]]),
        body_radii=np.array([1.0, 1.0]),
    )
    # 1.5 units away on the torus, but almost the whole world away on a plane.
    assert pick_creature(target, np.array([width - 1.0, 100.0])) == 7


def test_a_click_far_from_every_body_selects_nothing(config: GenomeConfig) -> None:
    _, _, frame = world(config)
    alone = dataclasses.replace(
        frame,
        uids=np.array([7], dtype=np.int64),
        positions=np.array([[10.0, 10.0]]),
        body_radii=np.array([1.0]),
    )
    assert pick_creature(alone, np.array([10.0 + 1.0 + PICK_REACH + 0.5, 10.0])) is None
    assert pick_creature(alone, np.array([10.0 + 1.0 + PICK_REACH - 0.5, 10.0])) == 7
    assert pick_creature(dataclasses.replace(alone, uids=alone.uids[:0]), np.zeros(2)) is None


def test_lobes_peak_at_each_receptor_and_close_at_the_edge(config: GenomeConfig) -> None:
    """At θ = μ_j, g_j = w(μ_j) = 1, because |μ_j| ≤ 70° leaves 20° > 15° to the edge (§5.2)."""
    sensors = config.sensors
    position = np.array([50.0, 60.0])
    # Light 1 and heading 0: the radius is r_vis and the angles are the ones of the receptors.
    lobes = retina_lobes(position, 0.0, 1.0, sensors)
    assert lobes.shape == (6, LOBE_SAMPLES, 2)
    theta = np.linspace(-math.pi / 2.0, math.pi / 2.0, LOBE_SAMPLES)
    reach = np.linalg.norm(lobes - position, axis=2)
    for j, mu in enumerate(sensors.receptor_mu_rad):
        nearest = int(np.argmin(np.abs(theta - mu)))
        sigma = sensors.receptor_sigma_rad[j]
        offset = (theta[nearest] - mu) / sigma
        assert reach[j, nearest] == pytest.approx(sensors.r_vis * math.exp(-0.5 * offset**2))
    assert reach[:, 0] == pytest.approx(np.zeros(6))
    assert reach[:, -1] == pytest.approx(np.zeros(6))


def test_lobes_shrink_at_night_and_turn_with_the_body(config: GenomeConfig) -> None:
    sensors = config.sensors
    origin = np.zeros(2)
    day = np.linalg.norm(retina_lobes(origin, 0.0, 1.0, sensors), axis=2).max()
    night = np.linalg.norm(retina_lobes(origin, 0.0, 0.0, sensors), axis=2).max()
    assert night == pytest.approx(day * sensors.m_noite)
    # Facing +y (angles grow to the right, §1.3), the rightmost receptor points to −x.
    turned = retina_lobes(origin, math.pi / 2.0, 1.0, sensors)
    tip = turned[5][np.argmax(np.linalg.norm(turned[5], axis=1))]
    assert tip[0] < 0.0


def test_nostrils_sit_ahead_of_the_body_at_both_sides(config: GenomeConfig) -> None:
    """δ = 2 · r_c + 1 at ±45° (§5.3): a body of radius 2 at heading 0 has δ = 5."""
    simulation, arrays, frame = world(config)
    uid = int(frame.uids[0])
    target = dataclasses.replace(
        frame,
        positions=np.vstack((np.array([100.0, 100.0]), frame.positions[1:])),
        headings=np.concatenate(([0.0], frame.headings[1:])),
        body_radii=np.concatenate(([2.0], frame.body_radii[1:])),
    )
    inspection = inspect_creature(arrays, target, uid, simulation.config)
    assert inspection is not None
    side = 5.0 / math.sqrt(2.0)
    # The left nostril is at −alpha_olf, which on screen coordinates is up (y smaller).
    assert inspection.nostrils[0].tolist() == pytest.approx([100.0 + side, 100.0 - side])
    assert inspection.nostrils[1].tolist() == pytest.approx([100.0 + side, 100.0 + side])


def test_inspection_reads_the_creature_and_its_inputs(config: GenomeConfig) -> None:
    simulation, arrays, frame = world(config)
    observed = simulation.observe()
    row = 3
    uid, slot = int(frame.uids[row]), int(observed[1][row])
    inspection = inspect_creature(arrays, frame, uid, simulation.config, observed)
    assert inspection is not None
    state = simulation.state
    assert inspection.uid == uid
    assert inspection.founder_id == int(state.founder_id[slot])
    assert inspection.energy == state.energy[slot]
    assert inspection.hydration == state.hydration[slot]
    assert inspection.mass == state.mass[slot]
    assert inspection.inputs is not None
    assert inspection.inputs.tolist() == observed[0][row].tolist()
    lines = panel_lines(inspection)
    assert lines[0].startswith(f"uid {uid}")
    assert any(line.startswith("vision food:") for line in lines)
    assert any(line.startswith("climate:") for line in lines)


def test_inspection_without_inputs_shows_only_the_body(config: GenomeConfig) -> None:
    simulation, arrays, frame = world(config)
    inspection = inspect_creature(arrays, frame, int(frame.uids[0]), simulation.config)
    assert inspection is not None
    assert inspection.inputs is None
    assert not any(line.startswith("vision") for line in panel_lines(inspection))


def test_a_creature_that_is_gone_has_no_inspection(config: GenomeConfig) -> None:
    simulation, arrays, frame = world(config)
    assert inspect_creature(arrays, frame, 10_000, simulation.config) is None
