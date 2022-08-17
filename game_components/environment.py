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