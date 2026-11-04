"""Drawing without a screen, and proof that watching changes nothing (ARQUITETURA §1.1, §2.1)."""

import dataclasses
from collections.abc import Iterator, Sequence

import numpy as np
import numpy.typing as npt
import pygame
import pytest

from genome.core.config import GenomeConfig
from genome.core.state_hash import state_hash
from genome.core.tick import Simulation
from genome.lab.controllers import FixedGenome
from genome.lab.ecology import make_simulation
from genome.lab.environments import fixed_controller, make_env
from genome.lab.viewer import (
    MAX_TICKS_PER_FRAME,
    ViewerState,
    apply_input,
    run_viewer,
    ticks_to_run,
)
from genome.viz.inspector import pick_creature
from genome.viz.renderer import (
    DAY_GROUND,
    NIGHT_GROUND,
    PANEL_PIXELS,
    WATER,
    WORLD_PIXELS,
    Click,
    Command,
    WorldRenderer,
)
from genome.viz.scene import WorldArrays, build_frame

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@pytest.fixture(autouse=True)
def no_screen(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    yield
    pygame.display.quit()


def small(config: GenomeConfig, ticks: int = 30) -> GenomeConfig:
    run = dataclasses.replace(config.run, max_population=30, max_ticks=ticks)
    world = dataclasses.replace(config.world, initial_population=12)
    training = dataclasses.replace(config.training, n=12, t_max=ticks)
    return dataclasses.replace(config, run=run, world=world, training=training)


def arrays_of(simulation: Simulation[FixedGenome]) -> dict[str, npt.NDArray[np.generic]]:
    return {name: array.copy() for name, array in simulation.named_arrays()}


def test_rgb_array_has_the_size_of_the_world_and_the_panel(config: GenomeConfig) -> None:
    chosen = small(config)
    pixels = WorldRenderer(chosen).render(arrays_of(make_simulation(chosen, 3, "immobile")))
    assert pixels is not None
    assert pixels.shape == (WORLD_PIXELS, WORLD_PIXELS + PANEL_PIXELS, 3)
    assert pixels.dtype == np.uint8


def test_the_window_of_a_big_world_keeps_its_size(config: GenomeConfig) -> None:
    chosen = small(config)
    world = dataclasses.replace(chosen.world, width=512, height=512)
    bigger = dataclasses.replace(chosen, world=world)
    assert WorldRenderer(bigger).world_point((768, 0)) is None
    point = WorldRenderer(bigger).world_point((384, 192))
    assert point is not None
    assert point.tolist() == [256.0, 128.0]


def test_the_ground_follows_the_light(config: GenomeConfig) -> None:
    chosen = small(config)
    simulation = make_simulation(chosen, 3, "immobile")
    renderer = WorldRenderer(chosen)
    arrays = arrays_of(simulation)
    # Tick 300 is day and tick 700 is night (t_dia = 600, t_crep = 50); no rain was drawn yet.
    day = renderer.render({**arrays, "tick": np.array([300], dtype=np.int64)})
    night = renderer.render({**arrays, "tick": np.array([700], dtype=np.int64)})
    assert day is not None
    assert night is not None
    colors_day = {tuple(c) for c in day[:, :WORLD_PIXELS].reshape(-1, 3).tolist()}
    colors_night = {tuple(c) for c in night[:, :WORLD_PIXELS].reshape(-1, 3).tolist()}
    raining = bool(arrays["climate.rain"][0])
    assert raining or DAY_GROUND in colors_day
    assert raining or NIGHT_GROUND in colors_night
    assert WATER in colors_day


def test_drawing_leaves_the_snapshot_untouched(config: GenomeConfig) -> None:
    chosen = small(config)
    simulation = make_simulation(chosen, 3, "immobile")
    arrays = arrays_of(simulation)
    before = state_hash(list(arrays.items()))
    observed = simulation.observe()
    selected = int(arrays["creatures.ids"][simulation.state.pool.active()][0])
    WorldRenderer(chosen).render(arrays, selected, observed, ("status",))
    assert state_hash(list(arrays.items())) == before
    assert state_hash(simulation.named_arrays()) == before


def test_a_selection_draws_the_lobes(config: GenomeConfig) -> None:
    chosen = small(config)
    simulation = make_simulation(chosen, 3, "immobile")
    arrays = arrays_of(simulation)
    renderer = WorldRenderer(chosen)
    plain = renderer.render(arrays)
    selected = int(arrays["creatures.ids"][simulation.state.pool.active()][0])
    marked = renderer.render(arrays, selected, simulation.observe())
    assert plain is not None
    assert marked is not None
    assert np.any(plain[:, :WORLD_PIXELS] != marked[:, :WORLD_PIXELS])
    # A selection of someone who is gone draws what the plain frame draws.
    gone = renderer.render(arrays, 10_000)
    assert gone is not None
    assert np.array_equal(gone, plain)


def test_keys_and_clicks_become_commands(config: GenomeConfig) -> None:
    renderer = WorldRenderer(small(config), "human")
    simulation = make_simulation(small(config), 3, "immobile")
    assert renderer.render(arrays_of(simulation)) is None
    pygame.event.clear()
    for key in (pygame.K_SPACE, pygame.K_PERIOD, pygame.K_PLUS, pygame.K_MINUS, pygame.K_ESCAPE):
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=key))
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_z))
    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(30, 60)))
    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(WORLD_PIXELS, 0)))
    pygame.event.post(pygame.event.Event(pygame.QUIT))
    found = renderer.poll()
    renderer.close()
    assert found[:5] == [
        Command.TOGGLE_PAUSE,
        Command.STEP,
        Command.FASTER,
        Command.SLOWER,
        Command.CLEAR_SELECTION,
    ]
    assert isinstance(found[5], Click)
    assert found[5].point.tolist() == [30.0 / 3.0, 60.0 / 3.0]
    assert found[6] is Command.QUIT
    assert len(found) == 7


def test_viewer_state_follows_the_commands() -> None:
    state = ViewerState()
    assert ticks_to_run(state) == 1
    state = apply_input(state, Command.FASTER)
    assert ticks_to_run(state) == 2
    state = apply_input(state, Command.TOGGLE_PAUSE)
    assert ticks_to_run(state) == 0
    state = apply_input(state, Command.STEP)
    assert ticks_to_run(state) == 1
    assert state.paused
    state = apply_input(ViewerState(), Command.SLOWER)
    assert state.ticks_per_frame == 1
    for _ in range(10):
        state = apply_input(state, Command.FASTER)
    assert state.ticks_per_frame == MAX_TICKS_PER_FRAME
    assert not apply_input(ViewerState(), Command.QUIT).running
    selected = ViewerState(selected=7)
    assert apply_input(selected, Command.CLEAR_SELECTION).selected is None


class ScriptedScreen:
    """Plays a list of inputs, one list per frame, and keeps what it was asked to draw."""

    def __init__(self, script: Sequence[list[Command | Click]]) -> None:
        self.script = list(script)
        self.ticks: list[int] = []
        self.selected: list[int | None] = []
        self.observed: list[bool] = []

    def render(
        self,
        state: WorldArrays,
        selected: int | None = None,
        observed: tuple[FloatArray, IntArray] | None = None,
        status: Sequence[str] = (),
    ) -> None:
        self.ticks.append(int(state["tick"][0]))
        self.selected.append(selected)
        self.observed.append(observed is not None)

    def poll(self) -> list[Command | Click]:
        return self.script.pop(0) if self.script else [Command.QUIT]

    def wait(self, frames_per_second: int) -> None:
        pass


def test_watching_a_world_changes_no_result(config: GenomeConfig) -> None:
    """Same ticks with and without the viewer, selection and sensors included (§1.1)."""
    chosen = small(config, ticks=20)
    direct = make_simulation(chosen, 3, "random")
    while not direct.finished:
        direct.step_batch()

    watched = make_simulation(chosen, 3, "random")
    first = watched.state.position[watched.state.pool.active()[0]]
    click = Click(first.copy())
    script: list[list[Command | Click]] = [[click], [Command.FASTER], [], [Command.FASTER]]
    screen = ScriptedScreen([*script, *[[] for _ in range(10)]])
    run_viewer(watched, screen)

    assert watched.finished
    assert state_hash(watched.named_arrays()) == state_hash(direct.named_arrays())
    assert any(screen.observed)


def test_the_viewer_pauses_steps_and_outlives_the_run(config: GenomeConfig) -> None:
    chosen = small(config, ticks=3)
    screen = ScriptedScreen(
        [
            [Command.TOGGLE_PAUSE],  # Frame 1: paused at once, nothing runs.
            [],
            [Command.STEP],  # One tick and still paused.
            [],
            [Command.TOGGLE_PAUSE],  # Runs to the end, and goes on showing it.
            [],
            [],
            [],
            [],
        ]
    )
    run_viewer(make_simulation(chosen, 3, "immobile"), screen)
    assert screen.ticks[:4] == [0, 0, 1, 1]
    assert screen.ticks[-1] == 3


def test_a_selection_ends_when_the_creature_dies(config: GenomeConfig) -> None:
    chosen = small(config, ticks=40)
    creatures = dataclasses.replace(chosen.creatures, founder_hydration_fraction=0.005)
    run = dataclasses.replace(chosen.run, terminate_on_extinction=False)
    dying = dataclasses.replace(chosen, creatures=creatures, run=run)
    simulation = make_simulation(dying, 3, "immobile")
    target = simulation.state.position[simulation.state.pool.active()[0]].copy()
    screen = ScriptedScreen([[Click(target)], *[[] for _ in range(39)]])
    run_viewer(simulation, screen)
    assert screen.selected[0] is not None
    assert screen.selected[-1] is None


def test_a_click_on_empty_ground_clears_the_selection(config: GenomeConfig) -> None:
    chosen = small(config, ticks=5)
    simulation = make_simulation(chosen, 3, "immobile")
    target = simulation.state.position[simulation.state.pool.active()[0]].copy()
    frame = build_frame(arrays_of(simulation), chosen)
    # A grid fine enough to hold a point out of reach of the 12 bodies of this world.
    grid = [np.array([x, y], dtype=np.float64) for x in range(0, 256, 8) for y in range(0, 256, 8)]
    empty = next(point for point in grid if pick_creature(frame, point) is None)
    screen = ScriptedScreen([[Click(target)], [], [Click(empty)]])
    run_viewer(simulation, screen)
    assert screen.selected[:2] == [screen.selected[0], screen.selected[0]]
    assert screen.selected[0] is not None
    assert screen.selected[2] is None


def test_escape_clears_the_selection_and_it_stays_cleared(config: GenomeConfig) -> None:
    chosen = small(config, ticks=5)
    simulation = make_simulation(chosen, 3, "immobile")
    target = simulation.state.position[simulation.state.pool.active()[0]].copy()
    screen = ScriptedScreen([[Click(target)], [], [Command.CLEAR_SELECTION], []])
    run_viewer(simulation, screen)
    assert screen.selected[1] is not None
    assert screen.selected[2:] == [None, None, None]


def test_the_environment_renders_through_the_renderer(config: GenomeConfig) -> None:
    chosen = small(config)
    env = make_env(chosen, render_mode="rgb_array")
    env.reset(seed=3, options={"controller": fixed_controller("immobile")})
    pixels = env.render()
    assert pixels.shape == (WORLD_PIXELS, WORLD_PIXELS + PANEL_PIXELS, 3)
    env.step({})
    assert env.render().shape == pixels.shape


def test_the_environment_shows_a_window_in_human_mode(config: GenomeConfig) -> None:
    env = make_env(small(config), render_mode="human")
    env.reset(seed=3, options={"controller": fixed_controller("immobile")})
    assert env.render() is None
