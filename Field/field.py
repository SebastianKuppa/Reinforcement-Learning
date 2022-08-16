import numpy as np


class Field:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.body = np.zeros((self.height, self.width))