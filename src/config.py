# src/config.py

BBOX_FULL = [28.765640, 40.098296, 29.419670, 40.349161]
IMAGE_WIDTH = 2668
IMAGE_HEIGHT = 1024

CLOUD_RGB = (192, 192, 192)
TRANSPARENT_RGBA = (0, 0, 0, 0)

COLOR_TO_SCORE = {
    (228, 0, 2): None,        # SOIL
    (255, 86, 0): 1.0,        # HIGH STRESS
    (107, 254, 147): 0.5,     # MODERATE
    (0, 239, 254): 0.2,       # LOW STRESS
    (0, 0, 143): 0.0          # NO STRESS
}
