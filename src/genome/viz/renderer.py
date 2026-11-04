"""Draws a snapshot with pygame-ce, in a window or to an array (ARQUITETURA §2.1, §2.2, ADR 0010).

The only module of the project that imports pygame, which comes with the extra ``viz``.
Everything it draws is read from the snapshot, and the simulation is never touched.
"""

import os
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, auto
from typing import Final, Literal

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import numpy.typing as npt
import pygame

from genome.core.config import GenomeConfig
from genome.viz.inspector import Inspection, inspect_creature, panel_lines
from genome.viz.scene import Frame, WorldArrays, build_frame

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
ByteArray = npt.NDArray[np.uint8]
Color = tuple[int, int, int]
RenderMode = Literal["human", "rgb_array"]

WORLD_PIXELS: Final = 768
PANEL_PIXELS: Final = 360
FONT_PIXELS: Final = 18
LINE_PIXELS: Final = 18
MARGIN_PIXELS: Final = 10
NIGHT_GROUND: Final[Color] = (24, 28, 44)
DAY_GROUND: Final[Color] = (112, 126, 88)
RAIN_SHADE: Final = 0.75
WATER: Final[Color] = (46, 104, 178)
#: Colour of a plant by its stack (§4.1), from young to fully grown.
PLANT_STACK_COLORS: Final[tuple[Color, ...]] = ((120, 176, 86), (70, 150, 60), (34, 112, 44))
PLANT_HALF_PIXELS: Final = 1
PANEL_COLOR: Final[Color] = (20, 22, 28)
TEXT_COLOR: Final[Color] = (226, 228, 232)
SELECTION_COLOR: Final[Color] = (255, 255, 255)
NOSTRIL_COLOR: Final[Color] = (255, 214, 64)
NOSTRIL_PIXELS: Final = 3
#: Translucent colours of the receptors, from the left of the field to the right (§5.2).
LOBE_COLORS: Final[tuple[tuple[int, int, int, int], ...]] = (
    (240, 96, 96, 70),
    (240, 170, 80, 70),
    (230, 230, 90, 70),
    (110, 220, 120, 70),
    (90, 190, 240, 70),
    (180, 130, 240, 70),
)
MIN_BODY_PIXELS: Final = 2
HEADING_FACTOR: Final = 1.6
SELECTION_EXTRA_PIXELS: Final = 3
WRAPS: Final = (-1, 0, 1)


class Command(Enum):
    """What the person at the window asked for; the laboratory decides what it means."""

    QUIT = auto()
    TOGGLE_PAUSE = auto()
    STEP = auto()
    FASTER = auto()
    SLOWER = auto()
    CLEAR_SELECTION = auto()


@dataclass(frozen=True, slots=True)
class Click:
    """A click on the world, in world units."""

    point: FloatArray


_KEYS: Final[dict[int, Command]] = {
    pygame.K_ESCAPE: Command.CLEAR_SELECTION,
    pygame.K_SPACE: Command.TOGGLE_PAUSE,
    pygame.K_PERIOD: Command.STEP,
    pygame.K_PLUS: Command.FASTER,
    pygame.K_EQUALS: Command.FASTER,
    pygame.K_KP_PLUS: Command.FASTER,
    pygame.K_MINUS: Command.SLOWER,
    pygame.K_KP_MINUS: Command.SLOWER,
    pygame.K_q: Command.QUIT,
}


class WorldRenderer:
    """Draws snapshots; implements the ``Renderer`` of ``genome.envs.parallel`` (ADR 0008)."""

    def __init__(self, config: GenomeConfig, mode: RenderMode = "rgb_array") -> None:
        """Prepare the canvas; a window opens only in ``human`` mode, at the first frame."""
        self._config = config
        self._mode = mode
        self._scale = WORLD_PIXELS / max(config.world.width, config.world.height)
        self._world_size = (
            round(config.world.width * self._scale),
            round(config.world.height * self._scale),
        )
        self._canvas = pygame.Surface((self._world_size[0] + PANEL_PIXELS, self._world_size[1]))
        self._window: pygame.Surface | None = None
        self._clock = pygame.time.Clock()
        pygame.font.init()
        self._font = pygame.font.Font(None, FONT_PIXELS)
        self._river_vertices: FloatArray | None = None
        self._river_mask: npt.NDArray[np.bool_] | None = None

    def render(
        self,
        state: WorldArrays,
        selected: int | None = None,
        observed: tuple[FloatArray, IntArray] | None = None,
        status: Sequence[str] = (),
    ) -> ByteArray | None:
        """Draw a snapshot.

        Args:
            state: The named arrays of the world.
            selected: ``uid`` of the creature to inspect.
            observed: ``Simulation.observe()``, to show its inputs.
            status: Extra lines for the top of the panel.

        Returns:
            The frame as RGB bytes of shape ``(height, width, 3)`` in ``rgb_array`` mode,
            and ``None`` in ``human`` mode, which shows it in the window.
        """
        frame = build_frame(state, self._config)
        inspection = None
        if selected is not None:
            inspection = inspect_creature(state, frame, selected, self._config, observed)
        self._draw(frame, inspection, status)
        if self._mode == "rgb_array":
            pixels: ByteArray = np.swapaxes(pygame.surfarray.array3d(self._canvas), 0, 1)
            return pixels
        self._show()
        return None

    def poll(self) -> list[Command | Click]:
        """What happened at the window since the last call, in order."""
        found: list[Command | Click] = []
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                found.append(Command.QUIT)
            elif event.type == pygame.KEYDOWN and event.key in _KEYS:
                found.append(_KEYS[event.key])
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                point = self.world_point(event.pos)
                if point is not None:
                    found.append(Click(point))
        return found

    def world_point(self, pixel: tuple[int, int]) -> FloatArray | None:
        """The world point under a pixel, or ``None`` if the pixel is on the panel."""
        if pixel[0] >= self._world_size[0] or pixel[1] >= self._world_size[1]:
            return None
        return np.array(pixel, dtype=np.float64) / self._scale

    def wait(self, frames_per_second: int) -> None:
        """Hold the window at a steady rate; the simulation does not depend on it (§1.1)."""
        self._clock.tick(frames_per_second)

    def close(self) -> None:
        """Close the window, if there is one."""
        if self._window is not None:
            pygame.display.quit()
            self._window = None

    def _show(self) -> None:
        if self._window is None:
            pygame.display.init()
            self._window = pygame.display.set_mode(self._canvas.get_size())
            pygame.display.set_caption("GENOME")
        self._window.blit(self._canvas, (0, 0))
        pygame.display.flip()
        pygame.event.pump()

    def _draw(self, frame: Frame, inspection: Inspection | None, status: Sequence[str]) -> None:
        self._canvas.fill(_ground(frame))
        self._draw_rivers(frame)
        self._draw_disks(frame.lake_centers, frame.lake_radii)
        self._draw_disks(frame.puddle_centers, frame.puddle_radii)
        self._draw_plants(frame)
        self._draw_creatures(frame)
        if inspection is not None:
            self._draw_selection(inspection, frame)
        self._draw_panel(frame, inspection, status)

    def _draw_rivers(self, frame: Frame) -> None:
        if frame.river_points.size == 0:
            return
        mask = self._rivers_mask(frame)
        pixels = pygame.surfarray.pixels3d(self._canvas)
        pixels[: mask.shape[0], : mask.shape[1]][mask] = WATER
        del pixels  # Unlocks the surface for the drawing that follows.

    def _rivers_mask(self, frame: Frame) -> npt.NDArray[np.bool_]:
        """Rivers as disks along their vertices; recomputed only for a new geography."""
        known = self._river_vertices
        if (
            self._river_mask is not None
            and known is not None
            and np.array_equal(known, frame.river_points)
        ):
            return self._river_mask
        reach = max(1, round(frame.river_radius * self._scale))
        steps = np.arange(-reach, reach + 1)
        dx, dy = np.meshgrid(steps, steps, indexing="ij")
        inside = dx * dx + dy * dy <= reach * reach
        width, height = self._world_size
        mask = np.zeros((width, height), dtype=np.bool_)
        centers = np.floor(frame.river_points * self._scale).astype(np.int64)
        xs = (centers[:, 0, np.newaxis] + dx[inside]) % width
        ys = (centers[:, 1, np.newaxis] + dy[inside]) % height
        mask[xs, ys] = True
        self._river_vertices = frame.river_points.copy()
        self._river_mask = mask
        return mask

    def _draw_disks(self, centers: FloatArray, radii: FloatArray) -> None:
        for center, radius in zip(centers.tolist(), radii.tolist(), strict=True):
            for shifted in self._copies(center):
                pygame.draw.circle(self._canvas, WATER, shifted, radius * self._scale)

    def _draw_plants(self, frame: Frame) -> None:
        if frame.plant_stacks.size == 0:
            return
        width, height = self._world_size
        centers = np.floor(frame.plant_positions * self._scale).astype(np.int64)
        colors = np.array(PLANT_STACK_COLORS, dtype=np.uint8)
        shades = colors[np.clip(frame.plant_stacks, 1, len(PLANT_STACK_COLORS)) - 1]
        steps = np.arange(-PLANT_HALF_PIXELS, PLANT_HALF_PIXELS + 1)
        dx, dy = (axis.ravel() for axis in np.meshgrid(steps, steps, indexing="ij"))
        pixels = pygame.surfarray.pixels3d(self._canvas)
        pixels[
            (centers[:, 0, np.newaxis] + dx) % width,
            (centers[:, 1, np.newaxis] + dy) % height,
        ] = shades[:, np.newaxis]
        del pixels  # Unlocks the surface for the drawing that follows.

    def _draw_creatures(self, frame: Frame) -> None:
        tips = np.column_stack((np.cos(frame.headings), np.sin(frame.headings)))
        for center, heading, radius, color in zip(
            frame.positions.tolist(),
            tips.tolist(),
            frame.body_radii.tolist(),
            frame.colors.tolist(),
            strict=True,
        ):
            pixels = max(MIN_BODY_PIXELS, radius * self._scale)
            for shifted in self._copies(center):
                tip = (
                    shifted[0] + HEADING_FACTOR * pixels * heading[0],
                    shifted[1] + HEADING_FACTOR * pixels * heading[1],
                )
                pygame.draw.circle(self._canvas, color, shifted, pixels)
                pygame.draw.line(self._canvas, color, shifted, tip)

    def _draw_selection(self, inspection: Inspection, frame: Frame) -> None:
        overlay = pygame.Surface(self._world_size, pygame.SRCALPHA)
        width, height = frame.size
        for wrap_x in WRAPS:
            for wrap_y in WRAPS:
                shift = np.array([wrap_x * width, wrap_y * height], dtype=np.float64)
                for lobe, color in zip(inspection.lobes, LOBE_COLORS, strict=True):
                    outline = np.vstack((inspection.position, lobe)) + shift
                    pygame.draw.polygon(overlay, color, (outline * self._scale).tolist())
        self._canvas.blit(overlay, (0, 0))
        center = inspection.position.tolist()
        ring = max(MIN_BODY_PIXELS, inspection.body_radius * self._scale) + SELECTION_EXTRA_PIXELS
        for shifted in self._copies(center):
            pygame.draw.circle(self._canvas, SELECTION_COLOR, shifted, ring, width=1)
        for nostril in inspection.nostrils.tolist():
            for shifted in self._copies(nostril):
                pygame.draw.circle(self._canvas, NOSTRIL_COLOR, shifted, NOSTRIL_PIXELS)

    def _draw_panel(
        self, frame: Frame, inspection: Inspection | None, status: Sequence[str]
    ) -> None:
        left = self._world_size[0]
        pygame.draw.rect(self._canvas, PANEL_COLOR, (left, 0, PANEL_PIXELS, self._world_size[1]))
        lines = [
            f"tick {frame.tick}",
            f"creatures {frame.uids.size}   plants {frame.plant_stacks.size}",
            f"light {frame.light:.2f}   {'rain' if frame.raining else 'dry'}"
            f"   humidity {frame.humidity:.2f}",
            *status,
            "",
        ]
        lines += panel_lines(inspection) if inspection is not None else ["click a creature"]
        for index, line in enumerate(lines):
            text = self._font.render(line, True, TEXT_COLOR)
            position = (left + MARGIN_PIXELS, MARGIN_PIXELS + index * LINE_PIXELS)
            self._canvas.blit(text, position)

    def _copies(self, center: Sequence[float]) -> list[tuple[float, float]]:
        """The pixel of a world point and its copies across the borders of the torus (§3.1)."""
        x, y = center[0] * self._scale, center[1] * self._scale
        width, height = self._world_size
        return [(x + i * width, y + j * height) for i in WRAPS for j in WRAPS]


def _ground(frame: Frame) -> Color:
    """The colour of the ground: darker at night and under rain (§3.6)."""
    shade = RAIN_SHADE if frame.raining else 1.0
    mixed = [
        round(shade * (night + frame.light * (day - night)))
        for night, day in zip(NIGHT_GROUND, DAY_GROUND, strict=True)
    ]
    return mixed[0], mixed[1], mixed[2]
