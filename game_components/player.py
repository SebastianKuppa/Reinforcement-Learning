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
        """
        Moves Player left or right
        :param field: The gaming field (np.array)
        :param direction: Keyboard action
        """
        valdir = {0: 0,
                  1: -1,
                  2: 1}
        direction = valdir[direction]
        next_x = self.x + (direction*self.speed)
        if not (next_x + self.width > field.width or next_x < 0):
            self.x += self.speed * direction
            self.stamina -= 1

    def change_width(self, action):
        """
        Changes the players width with up or down keys
        :param action: Keyboard input
        """
        val2act = {0: 0,
                   3: -1,
                   4: 1}
        action = val2act[action]
        new_width = self.width + action
        player_end = self.x + new_width
        if self.max_width >= new_width > 0 and player_end <= self.max_width:
            self.width = new_width
            self.body = np.ones((self.height, self.width))*self.body_unit
