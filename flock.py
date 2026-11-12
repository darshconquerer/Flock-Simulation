import pygame
import sys
import numpy as np
from calc import Flock

WIDTH = 1280
HEIGHT = 720
FPS_LIMIT = 64

BACKGROUND_COLOR = (0, 0, 0)
BOID_COLOR = (255, 255, 255)

START_BOID_COUNT = 500
MIN_BOID_COUNT = 10
MAX_BOID_COUNT = 1000
BOID_COUNT_STEP = 50

BOID_SHAPE = np.array([[6, 0.0],
                      [-2.0, 6.0],
                      [0.0, 0.0],
                      [-2.0, -6.0]
                      ])

def boid_corners(positions, headings):
    cos_angles = np.cos(headings)[:, None]
    sin_angles = np.sin(headings)[:, None]

    shape_x = BOID_SHAPE[:, 0]
    shape_y = BOID_SHAPE[:, 1]

    rotated_x = cos_angles * shape_x - sin_angles * shape_y
    rotated_y = sin_angles * shape_x - cos_angles * shape_y

    corners = np.stack((rotated_x, rotated_y), axis=2)
    return corners + positions[:, None, :]

def draw_flock(screen, flock):
    all_corners = boid_corners(flock.positions, flock.headings)
    for corners in all_corners:
        pygame.draw.polygon(screen, BOID_COLOR, corners)
    
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Chaos")
    clock = pygame.time.Clock()

    flock = Flock(START_BOID_COUNT, WIDTH, HEIGHT)
    paused = False

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    flock.reset()
                elif event.key == pygame.K_UP:
                    flock.set_boid_count(min(flock.boid_count + BOID_COUNT_STEP, MAX_BOID_COUNT))
                elif event.key == pygame.K_DOWN:
                    flock.set_boid_count(max(flock.boid_count - BOID_COUNT_STEP, MIN_BOID_COUNT))

        if not paused:
            flock.update()

        screen.fill(BACKGROUND_COLOR)
        draw_flock(screen, flock)

        pygame.display.update()
        clock.tick(FPS_LIMIT)

    pygame.quit()

if __name__ == "__main__":
    main()