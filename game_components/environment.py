import player
import field as f
import walls

from random import randint
from collections import deque
import pygame

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
    WALL_SPEED = 3

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
        self.field.update_field(walls=self.walls, player=self.player)

        observation = self.field.body / self.MAX_VAL

        return observation

    def print_text(self, WINDOW=None, text_cords=(0,0), center=False,
                   text="", color=(0,0,0), size=32):
        pygame.init()
        font = pygame.font.Font(name='freesansbold.ttf', size=size)
        text_to_print = font.render(text, True, color=color)
        textRect = text_to_print.get_rect()
        if center:
            textRect.center = text_cords
        else:
            textRect.x = text_cords[0]
            textRect.y = text_cords[1]
        WINDOW.blit(text_to_print, textRect)

    def step(self, action):
        global score_increased

        self.frames_counter += 1
        reward = 0

        if action in [1, 2]:
            self.player.move(field=self.field, direction=action)
        elif action in [3, 4]:
            self.player.change_width(action=action)

        if self.frames_counter % self.WALL_SPEED == 0:
            self.walls[-1].move()
            self.frames_counter = 0

        self.field.update_field(walls=self.walls, player=self.player)

        if ((self.walls[-1].y) == (self.player.y + self.player.height)) and score_increased ==False:
            reward += self.REWARD
            self.score += self.REWARD
            self.player.stamina += min(self.player.max_stamina, self.player.stamina+10)

            score_increased = True

            #  Lose Conditions :
            # C1 : The player hits a wall
            # C2 : Player's width was far thinner than hole's width
            # C3 : Player fully consumed its stamina (energy)
            lose_conds = [self.MAX_VAL in self.field.body,
                          (self.player.y == self.walls[-1].y and
                           self.player.width < self.walls[-1].hole_width-1),
                          self.player.stamina <= 0]

            if True in lose_conds:
                self.game_over = True
                reward = self.PUNISHMENT
                return self.field.body/self.MAX_VAL, reward, self.game_over

            if self.walls[-1].out_of_range:
                self.walls[-1] = walls.Wall(height=self.W_HEIGHT, width=self.WIDTH,
                                            hole_width=randint(self.MIN_H_WIDTH, self.MAX_H_WIDTH),
                                            field=self.field)
                score_increased = False

            return self.field.body/self.MAX_VAL, reward, self.game_over

    def render(self, WINDOW=None, human=False):
        if human:
            action = 0
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    self.game_over = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        action = 1
                    if event.key == pygame.K_RIGHT:
                        action = 2
                    if event.key == pygame.K_UP:
                        action = 4
                    if event.key == pygame.K_DOWN:
                        action = 3

            _, reward, self.game_over = self.step(action=action)

            WINDOW.fill(self.WHITE)
            self.field.update_field(walls=self.walls, player=self.player)

            for r in range(self.field.body.shape[0]):
                for c in range(self.field.body.shape[1]):
                    pygame.draw.rect(WINDOW,
                                     self.val2color[self.field.body[r][c]],
                                     (c*self.WIDTH_MUL, r*self.HEIGHT_MUL, self.WIDTH_MUL, self.HEIGHT_MUL))
                self.print_text(WINDOW=WINDOW, text_cords=(self.WINDOW_WIDTH // 2, int(self.WINDOW_HEIGHT * 0.1)),
                                text=str(self.score), color=self.RED, center=True)
                self.print_text(WINDOW=WINDOW, text_cords=(0, int(self.WINDOW_HEIGHT * 0.9)),
                                text=str(self.player.stamina), color=self.RED)

                pygame.display.update()

