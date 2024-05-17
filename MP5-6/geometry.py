# geometry.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Joshua Levine (joshua45@illinois.edu)
# Inspired by work done by James Gao (jamesjg2@illinois.edu) and Jongdeog Lee (jlee700@illinois.edu)

"""
This file contains geometry functions necessary for solving problems in MP5
"""

import numpy as np
from alien import Alien
from typing import List, Tuple
from copy import deepcopy


def does_alien_touch_wall(alien: Alien, walls: List[Tuple[int]]):
    """Determine whether the alien touches a wall

        Args:
            alien (Alien): Instance of Alien class that will be navigating our map
            walls (list): List of endpoints of line segments that comprise the walls in the maze in the format
                         [(startx, starty, endx, endx), ...]

        Return:
            True if touched, False if not
    """
    for i in range(len(walls)):
        head, tail = alien.get_head_and_tail()
        start = (walls[i][0], walls[i][1])
        end = (walls[i][2], walls[i][3])
        if (alien.is_circle()):
            if (point_segment_distance(alien.get_centroid(), (start,end)) <= alien.get_width()):
                return True
        elif (segment_distance((head,tail),(start,end)) <= alien.get_width()):
            return True
    return False


def is_alien_within_window(alien: Alien, window: Tuple[int]):
    """Determine whether the alien stays within the window

        Args:
            alien (Alien): Alien instance
            window (tuple): (width, height) of the window
    """
    if (alien.is_circle()):
        r = alien.get_width()
        pos = alien.get_centroid()

        if (pos[0] - r < 0) or (pos[0] + r > window[0]):
            return False
        if (pos[1] - r < 0) or (pos[1] + r > window[1]):
            return False
        return True
        
    d = alien.get_width()
    head, tail = alien.get_head_and_tail()
    if (head[0] - d <= 0) or (head[0] + d >= window[0]):
        return False
    if (head[1] - d <= 0) or (head[1] + d >= window[1]):
        return False
    if (tail[0] - d <= 0) or (tail[0] + d >= window[0]):
        return False
    if (tail[1] - d <= 0) or (tail[1] + d >= window[1]):
        return False
    return True


def is_point_in_polygon(point, polygon):
    """Determine whether a point is in a parallelogram.
    Note: The vertex of the parallelogram should be clockwise or counter-clockwise.

        Args:
            point (tuple): shape of (2, ). The coordinate (x, y) of the query point.
            polygon (tuple): shape of (4, 2). The coordinate (x, y) of 4 vertices of the parallelogram.
    """
    min_x = min(polygon[0][0], polygon[1][0], polygon[2][0], polygon[3][0])
    max_x = max(polygon[0][0], polygon[1][0], polygon[2][0], polygon[3][0])
    min_y = min(polygon[0][1], polygon[1][1], polygon[2][1], polygon[3][1])
    max_y = max(polygon[0][1], polygon[1][1], polygon[2][1], polygon[3][1])

    if not (min_x <= point[0] <= max_x and min_y <= point[1] <= max_y):
        return False

    ab = (polygon[0], polygon[1])
    bc = (polygon[1], polygon[2])
    cd = (polygon[2], polygon[3])
    da = (polygon[3], polygon[0])

    if (point_segment_distance(point, ab) == 0):
        return True
    if (point_segment_distance(point, bc) == 0):
        return True
    if (point_segment_distance(point, cd) == 0):
        return True
    if (point_segment_distance(point, da) == 0):
        return True

    seg = []
    dif = ((polygon[0][0]-point[0])*0.99 + point[0], (polygon[0][1]-point[1])*0.99 + point[1])
    seg.append((point, dif))
    dif = ((polygon[1][0]-point[0])*0.99 + point[0], (polygon[1][1]-point[1])*0.99 + point[1])
    seg.append((point, dif))
    dif = ((polygon[2][0]-point[0])*0.99 + point[0], (polygon[2][1]-point[1])*0.99 + point[1])
    seg.append((point, dif))
    dif = ((polygon[3][0]-point[0])*0.99 + point[0], (polygon[3][1]-point[1])*0.99 + point[1])
    seg.append((point, dif))

    for i in range(4):
        if do_segments_intersect(seg[i],ab):
            return False
        if do_segments_intersect(seg[i],bc):
            return False
        if do_segments_intersect(seg[i],cd):
            return False
        if do_segments_intersect(seg[i],da):
            return False
    return True


def does_alien_path_touch_wall(alien: Alien, walls: List[Tuple[int]], waypoint: Tuple[int, int]):
    """Determine whether the alien's straight-line path from its current position to the waypoint touches a wall

        Args:
            alien (Alien): the current alien instance
            walls (List of tuple): List of endpoints of line segments that comprise the walls in the maze in the format
                         [(startx, starty, endx, endx), ...]
            waypoint (tuple): the coordinate of the waypoint where the alien wants to move

        Return:
            True if touched, False if not
    """
    if (alien.is_circle()):
        seg = (alien.get_centroid(),waypoint)
        for i in range(len(walls)):
            wall = ((walls[i][0],walls[i][1]),(walls[i][2],walls[i][3]))
            if (segment_distance(seg, wall) <= alien.get_width()):
                return True
        return False
    
    w = alien.get_width()
    pos = alien.get_centroid()
    head, tail = alien.get_head_and_tail()
    delta_x = abs(head[0]-pos[0])
    delta_y = abs(head[1]-pos[1])
    wp_head = (waypoint[0]+delta_x, waypoint[1]-delta_y)
    wp_tail = (waypoint[0]-delta_x, waypoint[1]+delta_y)

    polygon = (head, wp_head, wp_tail, tail)
    for i in range(len(walls)):
        if (is_point_in_polygon((walls[i][0],walls[i][1]), polygon)):
            return True
        if (is_point_in_polygon((walls[i][2],walls[i][3]), polygon)):
            return True
        
        wall = ((walls[i][0],walls[i][1]),(walls[i][2],walls[i][3]))
        if (do_segments_intersect((pos,waypoint),wall)) or (segment_distance((pos,waypoint),wall) <= w):
            return True
        if (do_segments_intersect((head,wp_head),wall)) or (segment_distance((head,wp_head),wall) <= w):
            return True
        if (do_segments_intersect((tail,wp_tail),wall)) or (segment_distance((tail,wp_tail),wall) <= w):
            return True


    return False


def point_segment_distance(p, s):
    """Compute the distance from the point to the line segment.

        Args:
            p: A tuple (x, y) of the coordinates of the point.
            s: A tuple ((x1, y1), (x2, y2)) of coordinates indicating the endpoints of the segment.

        Return:
            Euclidean distance from the point to the line segment.
    """
    x1 = s[0][0]
    x2 = s[1][0]
    y1 = s[0][1]
    y2 = s[1][1]
    px = p[0]
    py = p[1]

    if (x1 == x2 and y1 == y2):
        return ((x1-px)**2 + (y1-py)**2)**(1/2)
    
    if y1 == y2:
        temp = px
        px = py
        py = -temp
        temp = x1
        x1 = y1
        y1 = -temp
        temp = x2
        x2 = y2
        y2 = -temp

    slope = (x2-x1) / (y1-y2)

    if (y1 > y2):
        x_big = x1
        y_big = y1
        x_small = x2
        y_small = y2
    else:
        x_big = x2
        y_big = y2
        x_small = x1
        y_small = y1

    if slope == 0:
        if (py > y_big):
            dist = ((px-x_big)**2 + (py-y_big)**2)**(1/2)
        elif (py < y_small):
            dist = ((px-x_small)**2 + (py-y_small)**2)**(1/2)
        else:
            a = (x2-x1, y2-y1)
            b = (px-x1, py-y1)
            dist = (a[0]*b[1] - a[1]*b[0]) / ((a[0]**2 + a[1]**2)**(1/2))
        return abs(dist)

    if (slope < 0 and x_big-(y_big-py)/slope < px) or (slope > 0 and x_big+(py-y_big)/slope > px):
        dist = ((px-x_big)**2 + (py-y_big)**2)**(1/2)
    elif (slope > 0 and x_small+(py-y_small)/slope < px) or (slope < 0 and x_small-(y_small-py)/slope > px):
        dist = ((px-x_small)**2 + (py-y_small)**2)**(1/2)
    else:
        a = (x2-x1, y2-y1)
        b = (px-x1, py-y1)
        dist = (a[0]*b[1] - a[1]*b[0]) / ((a[0]**2 + a[1]**2)**(1/2))

    return abs(dist)


def do_segments_intersect(s1, s2):
    """Determine whether segment1 intersects segment2.

        Args:
            s1: A tuple of coordinates indicating the endpoints of segment1.
            s2: A tuple of coordinates indicating the endpoints of segment2.

        Return:
            True if line segments intersect, False if not.
    """
    s1_x1 = s1[0][0]
    s1_x2 = s1[1][0]
    s1_y1 = s1[0][1]
    s1_y2 = s1[1][1]
    s2_x1 = s2[0][0]
    s2_x2 = s2[1][0]
    s2_y1 = s2[0][1]
    s2_y2 = s2[1][1]

    if (s1_x2-s1_x1 == 0) and (s2_x2-s2_x1 == 0):
        if (s1_x1 == s2_x1):
            if (min(s1_y1,s1_y2) <= s2_y1 <= max(s1_y1,s1_y2)) or (min(s1_y1,s1_y2) <= s2_y2 <= max(s1_y1,s1_y2)) or (min(s2_y1,s2_y2) <= s1_y2 <= max(s2_y1,s2_y2)):
                return True
        return False
    elif (s1_x2-s1_x1 == 0):
        slopeB = (s2_y2-s2_y1) / (s2_x2-s2_x1)
        intB = s2_y1 - slopeB*s2_x1
        if (min(s2_x1,s2_x2) <= s1_x1 <= max(s2_x1,s2_x2)):
            if min(s1_y1,s1_y2) <= slopeB*s1_x1 + intB <= max(s1_y1,s1_y2):
                return True
        return False
    elif (s2_x2-s2_x1 == 0):
        slopeA = (s1_y2-s1_y1) / (s1_x2-s1_x1)
        intA = s1_y1 - slopeA*s1_x1
        if (min(s1_x1,s1_x2) <= s2_x1 <= max(s1_x1,s1_x2)):
            if min(s2_y1,s2_y2) <= slopeA*s2_x1 + intA <= max(s2_y1,s2_y2):
                return True
        return False
    
    slopeA = (s1[1][1]-s1[0][1]) / (s1[1][0]-s1[0][0])
    slopeB = (s2[1][1]-s2[0][1]) / (s2[1][0]-s2[0][0])
    intA = s1[0][1] - slopeA*s1[0][0]
    intB = s2[0][1] - slopeB*s2[0][0]

    slopeDif = slopeB-slopeA
    intDif = intA-intB
    if slopeDif == 0:
        if intDif == 0:
            if (min(s1_x1,s1_x2) <= s2_x1 <= max(s1_x1,s1_x2)) or (min(s1_x1,s1_x2) <= s2_x2 <= max(s1_x1,s1_x2)) or (min(s2_x1,s2_x2) <= s1_x2 <= max(s2_x1,s2_x2)):
                return True
        return False
    intersect = intDif/slopeDif

    x_minA = min(s1[0][0], s1[1][0])
    x_maxA = max(s1[0][0], s1[1][0])
    x_minB = min(s2[0][0], s2[1][0])
    x_maxB = max(s2[0][0], s2[1][0])
    x_min = max(x_minA, x_minB)
    x_max = min(x_maxA, x_maxB)

    if (intersect >= x_min) and (intersect <= x_max):
        return True
    return False


def segment_distance(s1, s2):
    """Compute the distance from segment1 to segment2.  You will need `do_segments_intersect`.

        Args:
            s1: A tuple of coordinates indicating the endpoints of segment1.
            s2: A tuple of coordinates indicating the endpoints of segment2.

        Return:
            Euclidean distance between the two line segments.
    """
    if (do_segments_intersect(s1,s2)):
        return 0
    
    a = point_segment_distance(s1[0], s2)
    b = point_segment_distance(s1[1], s2)
    c = point_segment_distance(s2[0], s1)
    d = point_segment_distance(s2[1], s1)

    return min(a,b,c,d)


if __name__ == '__main__':

    from geometry_test_data import walls, goals, window, alien_positions, alien_ball_truths, alien_horz_truths, \
        alien_vert_truths, point_segment_distance_result, segment_distance_result, is_intersect_result, waypoints


    # Here we first test your basic geometry implementation
    def test_point_segment_distance(points, segments, results):
        num_points = len(points)
        num_segments = len(segments)
        for i in range(num_points):
            p = points[i]
            for j in range(num_segments):
                seg = ((segments[j][0], segments[j][1]), (segments[j][2], segments[j][3]))
                cur_dist = point_segment_distance(p, seg)
                assert abs(cur_dist - results[i][j]) <= 10 ** -3, \
                    f'Expected distance between {points[i]} and segment {segments[j]} is {results[i][j]}, ' \
                    f'but get {cur_dist}'


    def test_do_segments_intersect(center: List[Tuple[int]], segments: List[Tuple[int]],
                                   result: List[List[List[bool]]]):
        for i in range(len(center)):
            for j, s in enumerate([(40, 0), (0, 40), (100, 0), (0, 100), (0, 120), (120, 0)]):
                for k in range(len(segments)):
                    cx, cy = center[i]
                    st = (cx + s[0], cy + s[1])
                    ed = (cx - s[0], cy - s[1])
                    a = (st, ed)
                    b = ((segments[k][0], segments[k][1]), (segments[k][2], segments[k][3]))
                    if do_segments_intersect(a, b) != result[i][j][k]:
                        if result[i][j][k]:
                            assert False, f'Intersection Expected between {a} and {b}.'
                        if not result[i][j][k]:
                            assert False, f'Intersection not expected between {a} and {b}.'


    def test_segment_distance(center: List[Tuple[int]], segments: List[Tuple[int]], result: List[List[float]]):
        for i in range(len(center)):
            for j, s in enumerate([(40, 0), (0, 40), (100, 0), (0, 100), (0, 120), (120, 0)]):
                for k in range(len(segments)):
                    cx, cy = center[i]
                    st = (cx + s[0], cy + s[1])
                    ed = (cx - s[0], cy - s[1])
                    a = (st, ed)
                    b = ((segments[k][0], segments[k][1]), (segments[k][2], segments[k][3]))
                    distance = segment_distance(a, b)
                    assert abs(result[i][j][k] - distance) <= 10 ** -3, f'The distance between segment {a} and ' \
                                                                        f'{b} is expected to be {result[i]}, but your' \
                                                                        f'result is {distance}'


    def test_helper(alien: Alien, position, truths):
        alien.set_alien_pos(position)
        config = alien.get_config()

        touch_wall_result = does_alien_touch_wall(alien, walls)
        in_window_result = is_alien_within_window(alien, window)

        assert touch_wall_result == truths[
            0], f'does_alien_touch_wall(alien, walls) with alien config {config} returns {touch_wall_result}, ' \
                f'expected: {truths[0]}'
        assert in_window_result == truths[
            2], f'is_alien_within_window(alien, window) with alien config {config} returns {in_window_result}, ' \
                f'expected: {truths[2]}'


    def test_check_path(alien: Alien, position, truths, waypoints):
        alien.set_alien_pos(position)
        config = alien.get_config()

        for i, waypoint in enumerate(waypoints):
            path_touch_wall_result = does_alien_path_touch_wall(alien, walls, waypoint)

            assert path_touch_wall_result == truths[
                i], f'does_alien_path_touch_wall(alien, walls, waypoint) with alien config {config} ' \
                    f'and waypoint {waypoint} returns {path_touch_wall_result}, ' \
                    f'expected: {truths[i]}'

            # Initialize Aliens and perform simple sanity check.


    alien_ball = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Ball', window)
    test_helper(alien_ball, alien_ball.get_centroid(), (False, False, True))

    alien_horz = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Horizontal', window)
    test_helper(alien_horz, alien_horz.get_centroid(), (False, False, True))

    alien_vert = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Vertical', window)
    test_helper(alien_vert, alien_vert.get_centroid(), (True, False, True))

    edge_horz_alien = Alien((50, 100), [100, 0, 100], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Horizontal',
                            window)
    edge_vert_alien = Alien((200, 70), [120, 0, 120], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Vertical',
                            window)

    # Test validity of straight line paths between an alien and a waypoint
    test_check_path(alien_ball, (30, 120), (False, True, True), waypoints)
    test_check_path(alien_horz, (30, 120), (False, True, False), waypoints)
    test_check_path(alien_vert, (30, 120), (True, True, True), waypoints)

    centers = alien_positions
    segments = walls
    test_point_segment_distance(centers, segments, point_segment_distance_result)
    test_do_segments_intersect(centers, segments, is_intersect_result)
    test_segment_distance(centers, segments, segment_distance_result)

    for i in range(len(alien_positions)):
        test_helper(alien_ball, alien_positions[i], alien_ball_truths[i])
        test_helper(alien_horz, alien_positions[i], alien_horz_truths[i])
        test_helper(alien_vert, alien_positions[i], alien_vert_truths[i])

    # Edge case coincide line endpoints
    test_helper(edge_horz_alien, edge_horz_alien.get_centroid(), (True, False, False))
    test_helper(edge_horz_alien, (110, 55), (True, True, True))
    test_helper(edge_vert_alien, edge_vert_alien.get_centroid(), (True, False, True))

    print("Geometry tests passed\n")
