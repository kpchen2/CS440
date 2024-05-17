import numpy as np
import utils


class Agent:
    def __init__(self, actions, Ne=40, C=40, gamma=0.7, display_width=18, display_height=10):
        # HINT: You should be utilizing all of these
        self.actions = actions
        self.Ne = Ne  # used in exploration function
        self.C = C
        self.gamma = gamma
        self.display_width = display_width
        self.display_height = display_height
        self.reset()
        # Create the Q Table to work with
        self.Q = utils.create_q_table()
        self.N = utils.create_q_table()
        
    def train(self):
        self._train = True
        
    def eval(self):
        self._train = False

    # At the end of training save the trained model
    def save_model(self, model_path):
        utils.save(model_path, self.Q)
        utils.save(model_path.replace('.npy', '_N.npy'), self.N)

    # Load the trained model for evaluation
    def load_model(self, model_path):
        self.Q = utils.load(model_path)

    def reset(self):
        # HINT: These variables should be used for bookkeeping to store information across time-steps
        # For example, how do we know when a food pellet has been eaten if all we get from the environment
        # is the current number of points? In addition, Q-updates requires knowledge of the previously taken
        # state and action, in addition to the current state from the environment. Use these variables
        # to store this kind of information.
        self.points = 0
        self.s = None
        self.a = None
    
    def update_n(self, state, action):
        # TODO - MP11: Update the N-table. 
        if (state == None and action == None):
            return
        self.N[state + (action,)] += 1

    def update_q(self, s, a, r, s_prime):
        # TODO - MP11: Update the Q-table. 
        if (s == None and a == None):
            return
        alpha = self.C / (self.C + self.N[s + (a,)])
        m = self.Q[s_prime + (0,)]
        for i in range(1,len(self.actions)):
            if (self.Q[s_prime + (i,)] >= m):
                m = self.Q[s_prime + (i,)]
        self.Q[s + (a,)] += alpha*(r + self.gamma*m - self.Q[s + (a,)])


    def act(self, environment, points, dead):
        '''
        :param environment: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y, rock_x, rock_y] to be converted to a state.
        All of these are just numbers, except for snake_body, which is a list of (x,y) positions 
        :param points: float, the current points from environment
        :param dead: boolean, if the snake is dead
        :return: chosen action between utils.UP, utils.DOWN, utils.LEFT, utils.RIGHT

        Tip: you need to discretize the environment to the state space defined on the webpage first
        (Note that [adjoining_wall_x=0, adjoining_wall_y=0] is also the case when snake runs out of the playable board)
        '''
        s_prime = self.generate_state(environment)

        # TODO - MP12: write your function here

        return utils.RIGHT

    def generate_state(self, environment):
        '''
        :param environment: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y, rock_x, rock_y] to be converted to a state.
        All of these are just numbers, except for snake_body, which is a list of (x,y) positions 
        '''
        # TODO - MP11: Implement this helper function that generates a state given an environment 
        sx = environment[0]
        sy = environment[1]
        body = environment[2]
        fx = environment[3]
        fy = environment[4]
        rx = environment[5]
        ry = environment[6]

        invalid = -1

        food_on_same_x = 0
        food_on_left = 1
        food_on_right = 2
        food_on_same_y = 0
        food_on_top = 1
        food_on_bottom = 2

        no_obj_on_x = 0
        obj_on_left_right = 1
        obj_on_right = 2
        no_obj_on_y = 0
        obj_on_top_bottom = 1
        obj_on_bottom = 2

        body_on_top = 1
        body_on_bottom = 1
        body_on_left = 1
        body_on_right = 1

        # food_dir_x
        food_dir_x = invalid
        if (fx == sx):
            food_dir_x = food_on_same_x
        elif (fx < sx):
            food_dir_x = food_on_left
        elif (fx > sx):
            food_dir_x = food_on_right
        
        food_dir_y = invalid
        if (fy == sy):
            food_dir_y = food_on_same_y
        elif (fy < sy):
            food_dir_y = food_on_top
        elif (fy > sy):
            food_dir_y = food_on_bottom
        
        # adjoining_wall
        adjoining_wall_x = no_obj_on_x
        if (sx-2 == rx and sy == ry) or (sx-1 == 0):
            adjoining_wall_x = obj_on_left_right
        elif (sx+1 == rx and sy == ry) or (sx+2 == self.display_width):
            adjoining_wall_x = obj_on_right

        adjoining_wall_y = no_obj_on_y
        if (sy-1 == ry and ((sx == rx) or (sx == rx+1))) or (sy-1 == 0):
            adjoining_wall_y = obj_on_top_bottom
        elif (sy+1 == ry and ((sx == rx) or (sx == rx+1))) or (sy+2 == self.display_height):
            adjoining_wall_y = obj_on_bottom

        # adjoining_body
        adjoining_body_top = 0
        adjoining_body_bottom = 0
        adjoining_body_left = 0
        adjoining_body_right = 0
        for x, y in body:
            if (sy-1 == y and sx == x):
                adjoining_body_top = body_on_top
            if (sy+1 == y and sx == x):
                adjoining_body_bottom = body_on_bottom
            if (sx-1 == x and sy == y):
                adjoining_body_left = body_on_left
            if (sx+1 == x and sy == y):
                adjoining_body_right = body_on_right

        return (food_dir_x, food_dir_y, adjoining_wall_x, adjoining_wall_y, adjoining_body_top,
                adjoining_body_bottom, adjoining_body_left, adjoining_body_right)
