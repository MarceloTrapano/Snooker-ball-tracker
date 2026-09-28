import argparse
from dataclasses import dataclass, field

import cv2
import numpy as np
from numpy.typing import NDArray

from utils import ball_separator

Color = tuple[int, int, int]
Point = tuple[int, int]

WHITE: Color = (142, 142, 142)
BLUE: Color = (253, 128, 0)
GREEN: Color = (0, 255, 0)
PINK: Color = (180, 175, 236)
BROWN: Color = (0, 78, 149)
YELLOW: Color = (0, 241, 253)
BLACK: Color = (0, 0, 0)
RED: Color = (0, 0, 255)

DEFAULT_VIDEO = "video.mp4"

TABLE_X_MIN, TABLE_X_MAX = 125, 1175
TABLE_Y_MIN, TABLE_Y_MAX = 95, 615

MIN_BALL_AREA = 50
MAX_BALL_AREA = 400
MOVEMENT_THRESHOLD = 2
SETTLE_FRAMES = 50
RED_POINTS = 1

COLOUR_SEQUENCE = ("yellow", "green", "brown", "blue", "pink", "black")
POT_CHECK_ORDER = ("black", "yellow", "green", "brown", "blue", "pink")


def to_point(coordinate) -> Point:
    return int(coordinate[0]), int(coordinate[1])


def in_table(x: int, y: int) -> bool:
    return TABLE_X_MIN <= x <= TABLE_X_MAX and TABLE_Y_MIN <= y <= TABLE_Y_MAX


def find_ball_centers(mask, min_area=MIN_BALL_AREA, max_area=MAX_BALL_AREA) -> NDArray:
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for contour in contours:
        if not min_area < cv2.contourArea(contour) < max_area:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        x = int(moments["m10"] / moments["m00"])
        y = int(moments["m01"] / moments["m00"])
        if in_table(x, y):
            centers.append([x, y])
    return np.array(centers, dtype=int).reshape(-1, 2)


def draw_centers(frame, centers, color: Color) -> None:
    for center in centers:
        point = to_point(center)
        cv2.circle(frame, point, 15, color, 1)
        cv2.circle(frame, point, 1, color, 2)


def draw_direction(frame, start, end, color: Color) -> bool:
    line = end - start
    cv2.line(frame, to_point(end), to_point(end + line), color, 2)
    return bool(np.linalg.norm(line) > MOVEMENT_THRESHOLD)


def draw_score(frame, score: int) -> None:
    text = f"Score: {score}"
    position = (20, 40)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, text, position, font, 1, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, position, font, 1,
                (255, 255, 255), 2, cv2.LINE_AA)


@dataclass
class Ball:
    name: str
    color: Color
    points: int
    min_area: int = MIN_BALL_AREA
    active: bool = True
    position: NDArray = field(default_factory=lambda: np.zeros(2, dtype=int))
    centers: NDArray = field(
        default_factory=lambda: np.empty((0, 2), dtype=int))

    @property
    def missing(self) -> bool:
        return len(self.centers) == 0

    def detect(self, frame) -> None:
        mask = ball_separator(frame, self.name)
        self.centers = find_ball_centers(mask, self.min_area)

    def track(self, frame) -> bool:
        if self.missing:
            return False
        distances = np.linalg.norm(self.centers - self.position, axis=1)
        nearest = self.centers[np.argmin(distances)]
        draw_centers(frame, [nearest], self.color)
        moving = draw_direction(frame, self.position, nearest, self.color)
        self.position = nearest
        return moving


def create_colour_balls() -> dict[str, Ball]:
    balls = (
        Ball("blue", BLUE, 5, min_area=0),
        Ball("green", GREEN, 3, min_area=70),
        Ball("pink", PINK, 6),
        Ball("brown", BROWN, 4, min_area=150),
        Ball("yellow", YELLOW, 2),
        Ball("black", BLACK, 7),
    )
    return {ball.name: ball for ball in balls}


class SnookerTracker:
    def __init__(self) -> None:
        self.white = Ball("white", WHITE, 0)
        self.colours = create_colour_balls()
        self.red_centers: NDArray = np.empty((0, 2), dtype=int)
        self.reds_before_shot: int | None = None
        self.score = 0
        self.shot_scored = False
        self.still_frames = 0

    def process_frame(self, frame) -> None:
        self._detect_and_draw(frame)
        if self.reds_before_shot is None:
            self.reds_before_shot = len(self.red_centers) + 1

        if self.white.track(frame):
            self._on_movement()
        else:
            self._on_still()

    def _detect_and_draw(self, frame) -> None:
        self.white.detect(frame)
        for ball in self._active_colours():
            ball.detect(frame)
            ball.track(frame)
        self.red_centers = find_ball_centers(ball_separator(frame, "red"))
        draw_centers(frame, self.red_centers, RED)

    def _active_colours(self) -> list[Ball]:
        return [ball for ball in self.colours.values() if ball.active]

    def _on_movement(self) -> None:
        if self.shot_scored:
            self.shot_scored = False
            print("Move in progress")

    def _on_still(self) -> None:
        self.still_frames += 1
        if self.still_frames < SETTLE_FRAMES:
            return
        if not self.shot_scored:
            self.shot_scored = True
            self._finish_shot()
        self.still_frames = 0

    def _finish_shot(self) -> None:
        potted = self._find_potted_ball()
        if potted is not None:
            label, points = potted
            print(f"{label} ball potted")
            self.score += points
        print("Move finished")
        print(f"Current score: {self.score}")
        self.reds_before_shot = len(self.red_centers)
        self._update_active_colours()

    def _find_potted_ball(self) -> tuple[str, int] | None:
        if self.reds_before_shot > len(self.red_centers):
            return "Red", RED_POINTS
        for name in POT_CHECK_ORDER:
            ball = self.colours[name]
            if ball.active and ball.missing:
                return name.capitalize(), ball.points
        return None

    def _update_active_colours(self) -> None:
        previous_done = len(self.red_centers) == 0
        for name in COLOUR_SEQUENCE:
            ball = self.colours[name]
            if previous_done and ball.missing:
                ball.active = False
            previous_done = not ball.active


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", nargs="?", default=DEFAULT_VIDEO)
    args = parser.parse_args()

    video = cv2.VideoCapture(args.video)
    tracker = SnookerTracker()

    while video.isOpened():
        ok, frame = video.read()
        if not ok:
            break

        tracker.process_frame(frame)
        draw_score(frame, tracker.score)
        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
