"""The interactive viewer: the laboratory drives the world, ``genome.viz`` only draws it.

Implements ARQUITETURA §1.1 and §2.1. The world advances only here, in the same fixed
ticks as any other run, so watching it changes no result; ``ADR 0010`` has the details.
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Final, Protocol

import numpy as np
import numpy.typing as npt

from genome.core.tick import Simulation
from genome.lab.controllers import FixedGenome
from genome.viz.inspector import pick_creature
from genome.viz.renderer import Click, Command
from genome.viz.scene import WorldArrays, build_frame

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

MAX_TICKS_PER_FRAME: Final = 64
FRAMES_PER_SECOND: Final = 30


class ViewerScreen(Protocol):
    """What the viewer needs of a screen; ``WorldRenderer`` is the real one."""

    def render(
        self,
        state: WorldArrays,
        selected: int | None = None,
        observed: tuple[FloatArray, IntArray] | None = None,
        status: Sequence[str] = (),
    ) -> object:
        """Draw a snapshot."""
        ...

    def poll(self) -> list[Command | Click]:
        """What happened at the window since the last call."""
        ...

    def wait(self, frames_per_second: int) -> None:
        """Hold the frame rate."""
        ...


@dataclass(frozen=True, slots=True)
class ViewerState:
    """What the person at the window chose.

    Attributes:
        paused: Whether the world stands still.
        ticks_per_frame: Ticks run between two frames; it never changes the result.
        step_requested: Whether one tick was asked for while paused.
        running: False once the person quits.
        click: Point of a click not yet resolved into a selection.
        selected: ``uid`` of the creature under inspection.
    """

    paused: bool = False
    ticks_per_frame: int = 1
    step_requested: bool = False
    running: bool = True
    click: FloatArray | None = None
    selected: int | None = None


def apply_input(state: ViewerState, entry: Command | Click) -> ViewerState:
    """The state after one key or click."""
    if isinstance(entry, Click):
        return replace(state, click=entry.point)
    if entry is Command.QUIT:
        return replace(state, running=False)
    if entry is Command.TOGGLE_PAUSE:
        return replace(state, paused=not state.paused)
    if entry is Command.STEP:
        return replace(state, paused=True, step_requested=True)
    if entry is Command.FASTER:
        return replace(state, ticks_per_frame=min(MAX_TICKS_PER_FRAME, 2 * state.ticks_per_frame))
    if entry is Command.SLOWER:
        return replace(state, ticks_per_frame=max(1, state.ticks_per_frame // 2))
    if entry is Command.CLEAR_SELECTION:
        return replace(state, selected=None)
    return state


def ticks_to_run(state: ViewerState) -> int:
    """How many ticks this frame advances the world."""
    if state.step_requested:
        return 1
    return 0 if state.paused else state.ticks_per_frame


def run_viewer(simulation: Simulation[FixedGenome], screen: ViewerScreen) -> None:
    """Show a world until the person quits; it stays on screen after the run ends (§2.1)."""
    state = ViewerState()
    while state.running:
        for entry in screen.poll():
            state = apply_input(state, entry)
        for _ in range(ticks_to_run(state)):
            if simulation.finished:
                break
            simulation.step_batch()
        # A copy: whatever the screen does, it cannot reach the arrays of the world.
        arrays = {name: array.copy() for name, array in simulation.named_arrays()}
        frame = build_frame(arrays, simulation.config)
        selected = state.selected
        if state.click is not None:
            selected = pick_creature(frame, state.click)
        if selected is not None and selected not in frame.uids:
            selected = None
        status = [
            "paused" if state.paused else f"{state.ticks_per_frame} ticks per frame",
            "finished" if simulation.finished else "",
        ]
        screen.render(arrays, selected, _observed(simulation, selected), status)
        screen.wait(FRAMES_PER_SECOND)
        state = replace(state, step_requested=False, click=None, selected=selected)


def _observed(
    simulation: Simulation[FixedGenome], selected: int | None
) -> tuple[FloatArray, IntArray] | None:
    """The inputs of the next tick, only when someone looks; sensors draw nothing (§2.4)."""
    if selected is None:
        return None
    return simulation.observe()
