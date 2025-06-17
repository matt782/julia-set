import math

import numpy as np
from PIL import Image

# Constants
JULIA_SCREEN_WIDTH = 1024
JULIA_SCREEN_HEIGHT = 1024

JULIA_MAX_ITERATIONS = 255
JULIA_MAX_VALUE = 10

JULIA_REAL = 0.355524
JULIA_IMG = 0.337292
JULIA_CONSTANT = complex(JULIA_REAL, JULIA_IMG)

JULIA_RADIUS = 2
JULIA_CONVERSION_X_FACTOR = (2.0 * JULIA_RADIUS) / JULIA_SCREEN_WIDTH
JULIA_CONVERSION_Y_FACTOR = (-2.0 * JULIA_RADIUS) / JULIA_SCREEN_HEIGHT

JULIA_OUTPUT_FILE = "julia.png"

EPSILON = 1e-12


def julia_function(z):
    return z * z + JULIA_CONSTANT


def compute_julia():
    image = np.zeros((JULIA_SCREEN_HEIGHT, JULIA_SCREEN_WIDTH, 3), dtype=np.uint8)

    for i in range(JULIA_SCREEN_HEIGHT * JULIA_SCREEN_WIDTH):
        x = i % JULIA_SCREEN_WIDTH
        y = i // JULIA_SCREEN_WIDTH

        zx = x * JULIA_CONVERSION_X_FACTOR - JULIA_RADIUS
        zy = y * JULIA_CONVERSION_Y_FACTOR + JULIA_RADIUS
        z = complex(zx, zy)

        iteration = 0
        until iteration >= JULIA_MAX_ITERATIONS or abs(z) >= JULIA_MAX_VALUE:
            z = julia_function(z)
            iteration += 1

        if abs(z) > 1:
            d = (
                iteration
                - math.log2(math.log2(abs(z)))
                + math.log2(math.log2(JULIA_RADIUS))
            )
        else:
            d = 0
        d /= JULIA_MAX_ITERATIONS

        d = max(0, min(d, 1))  # clamp to [0, 1]

        r = int(pow(d, 1.0) * 255)
        g = int((1 - math.exp(-3.0 * d)) * 255)
        b = int((1 - math.exp(-6.5 * d)) * 255)

        image[y, x, 0] = r
        image[y, x, 1] = g
        image[y, x, 2] = b

    img = Image.fromarray(image, "RGB")
    img.save(JULIA_OUTPUT_FILE)


if __name__ == "__main__":
    compute_julia()
