import numpy as np
from random import randint


class Wall:
    def __init__(self, height=5, width=100, hole_width=20,
                 y=0, speed=1, field=None):
        self.height = height
        self.width = width
        self.hole_width = hole_width
        self.y = y
        self.speed = speed
        self.field = field
        self.body_unit = 1
        self.body = np.ones((self.height, self.width))*self.body_unit
        self.out_of_range = False
        self.create_hole()

    def create_hole(self):
        # hole = np.zeros((self.height, self.hole_width))
        hole_pos = randint(0, self.width - self.hole_width)
        self.body[:, hole_pos:hole_pos+self.hole_width] = 0

    def move(self):
        self.y += self.speed
        self.out_of_range = True if ((self.y + self.height) > self.field.height) else False
