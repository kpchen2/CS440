import heapq
# You do not need any other imports

def best_first_search(starting_state):
    '''
    Implementation of best first search algorithm

    Input:
        starting_state: an AbstractState object

    Return:
        A path consisting of a list of AbstractState states
        The first state should be starting_state
        The last state should have state.is_goal() == True
    '''
    # we will use this visited_states dictionary to serve multiple purposes
    # - visited_states[state] = (parent_state, distance_of_state_from_start)
    #   - keep track of which states have been visited by the search algorithm
    #   - keep track of the parent of each state, so we can call backtrack(visited_states, goal_state) and obtain the path
    #   - keep track of the distance of each state from start node
    #       - if we find a shorter path to the same state we can update with the new state 
    # NOTE: we can hash states because the __hash__/__eq__ method of AbstractState is implemented
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
    # Your code here ---------------

    ans = []
    cont = True
    visited_states[starting_state] = (starting_state, 0)

    while len(frontier) != 0:
        cur = frontier[0]
        heapq.heappop(frontier)

        # check is current state is goal
        if cur.is_goal():
            ans = backtrack(visited_states, cur)
            break

        # adding neighbors to frontier and visited
        neighbor_array = cur.get_neighbors()
        for neighbor in neighbor_array:
            if not (neighbor in visited_states.keys()):
            # if not ((neighbor in visited_states.keys()) and (visited_states[neighbor][1] < cur.dist_from_start+1)):
                heapq.heappush(frontier, neighbor)

            if (neighbor in visited_states.keys()):
                old_dist = visited_states[neighbor][1]
                if old_dist > cur.dist_from_start+1:
                    visited_states[neighbor] = (cur, cur.dist_from_start+1)
            else:
                visited_states[neighbor] = (cur, cur.dist_from_start+1)
        if cont:
            cont = False

    # ------------------------------
    
    # if you do not find the goal return an empty list
    return ans

# TODO(III): implement backtrack method, to be called by best_first_search upon reaching goal_state
# Go backwards through the pointers in visited_states until you reach the starting state
# NOTE: the parent of the starting state is None
def backtrack(visited_states, goal_state):
    path = []
    # Your code here ---------------

    cur = goal_state
    if visited_states[goal_state][1] > 0:
        while True:
            path.append(cur)
            cur = visited_states[cur][0]
            if visited_states[cur][1] == 0:
                break
        path.append(visited_states[cur][0])
    else:
        path.append(goal_state)
    path.reverse()

    # ------------------------------
    return path