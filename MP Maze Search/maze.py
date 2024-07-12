# maze.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
# 
# Created by Joshua Levine (joshua45@illinois.edu) and Jiaqi Gun
"""
This file contains the Maze class, which reads in a maze file and creates
a representation of the maze that is exposed through a simple interface.
"""

import copy
from state import MazeState, euclidean_distance
from geometry import does_alien_path_touch_wall, does_alien_touch_wall


class MazeError(Exception):
    pass


class NoStartError(Exception):
    pass


class NoObjectiveError(Exception):
    pass


class Maze:
    def __init__(self, alien, walls, waypoints, goals, move_cache={}, k=5, use_heuristic=True):
        """Initialize the Maze class, which will be navigated by a crystal alien

        Args:
            alien: (Alien), the alien that will be navigating our map
            walls: (List of tuple), List of endpoints of line segments that comprise the walls in the maze in the format
                        [(startx, starty, endx, endx), ...]
            waypoints: (List of tuple), List of waypoint coordinates in the maze in the format of [(x, y), ...]
            goals: (List of tuple), List of goal coordinates in the maze in the format of [(x, y), ...]
            move_cache: (Dict), caching whether a move is valid in the format of
                        {((start_x, start_y, start_shape), (end_x, end_y, end_shape)): True/False, ...}
            k (int): the number of waypoints to check when getting neighbors
        """
        self.k = k
        self.alien = alien
        self.walls = walls

        self.states_explored = 0
        self.move_cache = move_cache
        self.use_heuristic = use_heuristic

        self.__start = (*alien.get_centroid(), alien.get_shape_idx())
        self.__objective = tuple(goals)

        # Waypoints: the alien must move between waypoints (goal is a special waypoint)
        # Goals are also viewed as a part of waypoints
        self.__waypoints = waypoints + goals
        self.__valid_waypoints = self.filter_valid_waypoints()
        self.__start = MazeState(self.__start, self.get_objectives(), 0, self, self.use_heuristic)

        # self.__dimensions = [len(input_map), len(input_map[0]), len(input_map[0][0])]
        # self.__map = input_map

        if not self.__start:
            # raise SystemExit
            raise NoStartError("Maze has no start")

        if not self.__objective:
            raise NoObjectiveError("Maze has no objectives")

        if not self.__waypoints:
            raise NoObjectiveError("Maze has no waypoints")

    def is_objective(self, waypoint):
        """"
        Returns True if the given position is the location of an objective
        """
        return waypoint in self.__objective

    # Returns the start position as a tuple of (row, col, level)
    def get_start(self):
        assert (isinstance(self.__start, MazeState))
        return self.__start

    def set_start(self, start):
        """
        Sets the start state
        start (MazeState): a new starting state
        return: None
        """
        self.__start = start

    # Returns the dimensions of the maze as a (num_row, num_col, level) tuple
    # def get_dimensions(self):
    #     return self.__dimensions

    # Returns the list of objective positions of the maze, formatted as (x, y, shape) tuples
    def get_objectives(self):
        return copy.deepcopy(self.__objective)

    def get_waypoints(self):
        return self.__waypoints

    def get_valid_waypoints(self):
        return self.__valid_waypoints

    def set_objectives(self, objectives):
        self.__objective = objectives

    # TODO VI
    def filter_valid_waypoints(self):
        """Filter valid waypoints on each alien shape

            Return:
                A dict with shape index as keys and the list of waypoints coordinates as values
        """
        # shapes: 'Horizontal', 'Ball', 'Vertical'
        valid_waypoints = {i: [] for i in range(len(self.alien.get_shapes()))}

        dummy = self.create_new_alien(0,0,0)

        # test horizontal
        dummy.set_alien_shape('Horizontal')
        for wp in self.get_waypoints():
            dummy.set_alien_pos(wp)
            if not does_alien_touch_wall(dummy, self.walls):
                valid_waypoints[0].append(wp)

        # test ball
        dummy.set_alien_shape('Ball')
        for wp in self.get_waypoints():
            dummy.set_alien_pos(wp)
            if not does_alien_touch_wall(dummy, self.walls):
                valid_waypoints[1].append(wp)

        # test vertical
        dummy.set_alien_shape('Vertical')
        for wp in self.get_waypoints():
            dummy.set_alien_pos(wp)
            if not does_alien_touch_wall(dummy, self.walls):
                valid_waypoints[2].append(wp)

        return valid_waypoints

    # TODO VI
    def get_nearest_waypoints(self, cur_waypoint, cur_shape):
        """Find the k nearest valid neighbors to the cur_waypoint from a list of 2D points.
            Args:
                cur_waypoint: (x, y) waypoint coordinate
                cur_shape: shape index
            Return:
                the k valid waypoints that are closest to waypoint
        """
        nearest_neighbors = []

        # get nearest valid waypoints for current shape
        valid_wp_dict = self.filter_valid_waypoints()
        valid_wp = valid_wp_dict[cur_shape]

        # check if valid waypoints make valid paths
        temp = []
        for wp in valid_wp:
            if self.is_valid_move((cur_waypoint[0], cur_waypoint[1], cur_shape), (wp[0], wp[1], cur_shape)):
                if (cur_waypoint != wp):
                    temp.append(wp)
        valid_wp = temp
        
        # get path length for each path
        tups = []
        for i in range(len(valid_wp)):
            dist = ((cur_waypoint[0]-valid_wp[i][0])**2 + (cur_waypoint[1]-valid_wp[i][1])**2)**(0.5)
            tups.append((dist, i))

        # sort by path length
        tups = sorted(tups)
        for i in range(min(self.k,len(tups))):
            nearest_neighbors.append(valid_wp[tups[i][1]])

        return nearest_neighbors

    def create_new_alien(self, x, y, shape_idx):
        alien = copy.deepcopy(self.alien)
        alien.set_alien_config([x, y, self.alien.get_shapes()[shape_idx]])
        return alien

    # TODO VI
    def is_valid_move(self, start, end):
        """Check if the position of the waypoint can be reached by a straight-line path from the current position
            Args:
                start: (start_x, start_y, start_shape_idx)
                end: (end_x, end_y, end_shape_idx)
            Return:
                True if the move is valid, False otherwise
        """
        dummy = self.create_new_alien(0,0,0)

        # check shape/movement edge cases
        if (start[2] > 2 or start[2] < 0):
            return False
        if (end[2] > 2 or end[2] < 0):
            return False
        if (abs(start[2]-end[2]) > 1):
            return False
        if (start[2] != end[2] and (start[0],start[1]) != (end[0],end[1])):
            return False
        if (start[2] == end[2] and (start[0],start[1]) == (end[0],end[1])):
            return False
        
        # change to current shape
        if (start[2] != end[2]):
            dummy.set_alien_shape('Ball')
            if (end[2] == 0):
                dummy.set_alien_shape('Horizontal')
            elif (end[2] == 2):
                dummy.set_alien_shape('Vertical')
            # check valid shape
            dummy.set_alien_pos((start[0],start[1]))
            if does_alien_touch_wall(dummy, self.walls):
                return False
            return True

        # change to current shape
        dummy.set_alien_shape('Ball')
        if (end[2] == 0):
            dummy.set_alien_shape('Horizontal')
        elif (end[2] == 2):
            dummy.set_alien_shape('Vertical')
        # check valid shape
        dummy.set_alien_pos((start[0],start[1]))
        if does_alien_path_touch_wall(dummy, self.walls, (end[0],end[1])):
            return False

        return True



    def get_neighbors(self, x, y, shape_idx):
        """Returns list of neighboring squares that can be moved to from the given coordinate
            Args:
                x: query x coordinate
                y: query y coordinate
                shape_idx: query shape index
            Return:
                list of possible neighbor positions, formatted as (x, y, shape) tuples.
        """
        self.states_explored += 1

        nearest = self.get_nearest_waypoints((x, y), shape_idx)
        neighbors = [(*end, shape_idx) for end in nearest]
        for end in [(x, y, shape_idx - 1), (x, y, shape_idx + 1)]:
            start = (x, y, shape_idx)
            if self.is_valid_move(start, end):
                neighbors.append(end)
        return neighbors
