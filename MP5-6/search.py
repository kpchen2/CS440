# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains search functions.
"""
# Search should return the path and the number of states explored.
# The path should be a list of tuples in the form (alpha, beta, gamma) that correspond
# to the positions of the path taken by your search algorithm.
# Number of states explored should be a number.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,astar)
# You may need to slight change your previous search functions in MP1 since this is 3-d maze

from collections import deque
import heapq


# Search should return the path and the number of states explored.
# The path should be a list of MazeState objects that correspond
# to the positions of the path taken by your search algorithm.
# Number of states explored should be a number.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (astar)
# You may need to slight change your previous search functions in MP2 since this is 3-d maze


def search(maze, searchMethod):
    return {
        "astar": astar,
    }.get(searchMethod, [])(maze)


# TODO: VI
def astar(maze):

    starting_state = maze.get_start()
    # starting_state = ((init[0],init[1],init[2]), maze.get_objectives, 0, maze)

    visited_states = {starting_state: (None, 0)}

    # The frontier is a priority queue
    # You can pop from the queue using "heapq.heappop(frontier)"
    # You can push onto the queue using "heapq.heappush(frontier, state)"
    # NOTE: states are ordered because the __lt__ method of AbstractState is implemented
    frontier = []
    heapq.heappush(frontier, starting_state)
    
    # TODO(III): implement the rest of the best first search algorithm
    # HINTS:
    #   - add new states to the frontier by calling state.get_neighbors()
    #   - check whether you've finished the search by calling state.is_goal()
    #       - then call backtrack(visited_states, state)...
    #   - you can reuse the search code from mp3...
    # Your code here ---------------

    ans = []
    visited_states[starting_state] = (starting_state, 0)

    while len(frontier) != 0:
        cur = frontier[0]
        heapq.heappop(frontier)

        # check is current state is goal
        if cur.is_goal():
            ans = backtrack(visited_states, cur)
            break
        
        cur.dist_from_start = visited_states[cur][1]

        # adding neighbors to frontier and visited
        neighbor_array = cur.get_neighbors()
        for neighbor in neighbor_array:
            dist = dist_helper(cur.state,neighbor.state)
            if not (neighbor in visited_states.keys()):
                heapq.heappush(frontier, neighbor)
                visited_states[neighbor] = (cur, cur.dist_from_start+dist)
            else:
                if visited_states[neighbor][1] > cur.dist_from_start+dist:
                    visited_states[neighbor] = (cur, cur.dist_from_start+dist)
                    heapq.heappush(frontier, neighbor)

    # ------------------------------
    
    # if you do not find the goal return an empty list
    if (len(ans) == 0):
        return None
    return ans


# euclidean distance helper function
def dist_helper(a, b):
    if (a[2] != b[2]):
        return 10
    
    dist = ((a[0] - b[0])**2 + (a[1] - b[1])**2)**(0.5)
    return dist


# Go backwards through the pointers in visited_states until you reach the starting state
# NOTE: the parent of the starting state is None
# TODO: VI
def backtrack(visited_states, current_state):
    
    path = []
    # Your code here ---------------

    cur = current_state
    if visited_states[current_state][1] > 0:
        while True:
            path.append(cur)
            # print("CURRENT", cur)
            # print("NEXT",visited_states[cur][0])
            # print("DIST", visited_states[cur][1])
            # print("")
            cur = visited_states[cur][0]
            if visited_states[cur][1] == 0:
                break
        path.append(visited_states[cur][0])
        # print("CURRENT", cur)
        # print("")
    else:
        path.append(current_state)
    path.reverse()

    # ------------------------------
    return path
