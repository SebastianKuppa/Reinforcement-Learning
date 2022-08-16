import numpy as np


class Player:
    def __init__(self, height=5, max_width=10, width=2,
                 x=0, y=0, speed=2):
        self.height = height
        self.width = width
        self.max_width = max_width
        self.x = x
        self.y = y
        self.speed = speed
        self.body_unit = 2
        self.body = np.ones((self.height, self.width))*self.body_unit
        self.stamina = 20
        self.max_stamina = 20

    def move(self, field, direction=0):
        '''
        Moves the player :
         - No change          = 0
         - left, if direction  = 1
         - right, if direction = 2
        '''
        valdir = {0: 0,
                  1: -1,
                  2: 1}
        direction = valdir[direction]