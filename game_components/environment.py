import player
import field as f
import walls

from random import randint
from collections import deque

class Environment:
    P_HEIGHT = 2  # player height
    F_HEIGHT = 20  # field height
    W_HEIGHT = 2  # walls height
    WIDTH = 10  # walls and field width
    MIN_H_WIDTH = 2  # min hole width
    MAX_H_WIDTH = 6  # max hole width
    MIN_P_WIDTH = 2  # min player width
    MAX_P_WIDTH = 6  # max player width
    HEIGHT_MUL = 30  # pygame multiplier for drawing blocks
    WIDTH_MUL = 40  # pygame multiplier for drawing blocks
    WINDOW_HEIGHT = (F_HEIGHT + 1) * HEIGHT_MUL  # pygame window
    WINDOW_WIDTH = (WIDTH) * WIDTH_MUL  # pygame window

    ENVIRONMENT_SHAPE = (F_HEIGHT, WIDTH, 1)
    ACTION_SPACE = [0, 1, 2, 3, 4]
    ACTION_SPACE_SIZE = len(ACTION_SPACE)
    PUNISHMENT = -100
    REWARD = 10
    score = 0  # starting score

    MOVE_WALL_EVERY = 4  # wall frame move
    MOVE_PLAYER_EVERY = 1  # player frame move.
    frames_counter = 0 # number of current frames

    def __init__(self):
        self.BLACK = (25, 25, 25)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 80, 80)
        self.BLUE = (80, 80, 255)
        self.field = self.walls = self.player = None
        self.current_state = self.reset()
        self.val2color = {0:                        self.WHITE,
                          self.walls[0].body_unit:  self.BLACK,
                          self.player.body_unit:    self.BLACK,
                          self.MAX_VAL:             self.RED}

    def reset(self):
        self.score = 0
        self.frames_counter = 0
        self.game_over = False

        self.field = f.Field(width=self.WIDTH, height=self.F_HEIGHT)
        w1 = walls.Wall(height=self.W_HEIGHT, width=self.WIDTH,
                        hole_width=randint(self.MIN_H_WIDTH, self.MAX_H_WIDTH),
                        field=self.field)
        self.walls = deque([w1])
        p_width = randint(self.MIN_P_WIDTH, self.MAX_P_WIDTH)
        self.player = player.Player(height=self.P_HEIGHT, max_width=self.WIDTH,
                                    width=p_width, x=randint(0, self.field.width-p_width),
                                    y=int(self.field.height*.7), speed=1)
        self.MAX_VAL = self.player.body_unit + w1.body_unit
        # self.update_field

        return 0