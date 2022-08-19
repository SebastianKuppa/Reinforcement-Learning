import numpy as np


class Field:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.body = np.zeros((self.height, self.width))

    def update_field(self, walls, player):
        try:
            # clear field
            self.body = np.zeros((self.height, self.width))
            for wall in walls:
                if not wall.out_of_range:
                    self.body[wall.y:min(wall.y+wall.height, self.height), :] = wall.body
            # player
            self.body[player.y:player.y+player.height,
                      player.x:player.x+player.width] += player.body
        except:
            pass
