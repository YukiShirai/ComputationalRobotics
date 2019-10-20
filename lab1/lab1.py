# EE Computational Robotics: lab 1
#
# Problem 1

# Problem 1(a), (b)

import numpy as np
import matplotlib.pyplot as plt
import math
import time

# Define constants

L = 8
W = 8

# State space: 8 * 8 * 12
s = np.empty([8, 8, 12])
S = []
for i in range(8):
    for j in range(8):
        for k in range(12):
            temp = np.array([i, j, k])
            S.append([temp])
# S = [i,j,k] describes the state space. i represents x position, j represents y position, and k represents heading angle.
S = np.array(S)
# A = [i,j] represents action of a robot, where i == +1/-1 represents forward/backward linear motion.
# j == +1/-1 represents clockwise/counter-clockwise rotational motion.
A = np.array([[1, 1, 1, -1, -1, -1, 0], [-1, 0, 1, -1, 0, 1, 0]])
nA = len(A[0])
print('The size of the state space is', s.size)
print('The size of the action space is', len(A[0]))


# Problem 1(c)
def calc_prob(pe, s, a, s_prime):
    """
    Calculate the transition probability
    :param pe:error probability
    :param s:state
    :param a:action
    :param s_prime:transit state
    :return:transition probability

    # Idea: create array that deal with all possible state transitions after action takes.
    # Then, calculate the possibility based on the current & future state and action.

    # P_trans [i] = [(current intended action), (error action left), (error action right)]
    # i: heading angle
    # current intended action = [(intended linear movement), (heading direction), possibility of action]
    linear movement: +x [1,0], -x [-1,0], +y [0,1], -y [0,-1]
    heading direction: 0:1:11


    """

    P_trans = {}
    Prob_tmp = {}
    P_trans[0] = [([0, 1], 0, 1 - 2 * pe), ([0, 1], 11, pe), ([0, 1], 1, pe)]
    P_trans[1] = [([0, 1], 1, 1 - 2 * pe), ([0, 1], 0, pe), ([1, 0], 2, pe)]
    P_trans[2] = [([1, 0], 2, 1 - 2 * pe), ([0, 1], 1, pe), ([1, 0], 3, pe)]
    P_trans[3] = [([1, 0], 3, 1 - 2 * pe), ([1, 0], 2, pe), ([1, 0], 4, pe)]
    P_trans[4] = [([1, 0], 4, 1 - 2 * pe), ([1, 0], 3, pe), ([0, -1], 5, pe)]
    P_trans[5] = [([0, -1], 5, 1 - 2 * pe), ([1, 0], 4, pe), ([0, -1], 6, pe)]
    P_trans[6] = [([0, -1], 6, 1 - 2 * pe), ([0, -1], 5, pe), ([0, -1], 7, pe)]
    P_trans[7] = [([0, -1], 7, 1 - 2 * pe), ([0, -1], 6, pe), ([-1, 0], 8, pe)]
    P_trans[8] = [([-1, 0], 8, 1 - 2 * pe), ([0, -1], 7, pe), ([-1, 0], 9, pe)]
    P_trans[9] = [([-1, 0], 9, 1 - 2 * pe), ([-1, 0], 8, pe), ([-1, 0], 10, pe)]
    P_trans[10] = [([-1, 0], 10, 1 - 2 * pe), ([-1, 0], 9, pe), ([0, 1], 11, pe)]
    P_trans[11] = [([0, 1], 11, 1 - 2 * pe), ([-1, 0], 10, pe), ([0, 1], 0, pe)]

    if a[0] == 1:  # Forward motion
        for P_tmp in P_trans[s[2]]:
            x_goal = s[0] + P_tmp[0][0]
            y_goal = s[1] + P_tmp[0][1]
            h_goal = (a[1] + P_tmp[1]) % 12
            # At edges of the grids, the robot can only rotate:
            if x_goal < 0 or x_goal > L - 1:
                x_goal = s[0]
            if y_goal < 0 or y_goal > W - 1:
                y_goal = s[1]

            Prob_tmp[x_goal, y_goal, h_goal] = P_tmp[2]

    if a[0] == -1:  # Backward motion
        for P_tmp in P_trans[s[2]]:
            x_goal = s[0] - P_tmp[0][0]
            y_goal = s[1] - P_tmp[0][1]
            h_goal = (a[1] + P_tmp[1]) % 12
            # At edges of the grids, the robot can only rotate:
            if x_goal < 0 or x_goal > L - 1:
                x_goal = s[0]
            if y_goal < 0 or y_goal > W - 1:
                y_goal = s[1]

            Prob_tmp[x_goal, y_goal, h_goal] = P_tmp[2]

    if a[0] == 0:  # No linear motion, i.e. no error
        x_goal = s[0]
        y_goal = s[1]
        h_goal = (a[1] + s[2]) % 12
        Prob_tmp[x_goal, y_goal, h_goal] = 1

    if s_prime in Prob_tmp.keys():
        return Prob_tmp[s_prime]
    else:
        return 0


# Test for Problem 1(c)
# s = (0,3,4)
# s_prime = (1,3,4)
# pe= 0.2
# for i in range(len(A[0])):
#     a = A[:,i]
#     pr = calc_prob(pe,s,a,s_prime)
#     print("action:",a)
#     print("pr:",pr)


# Problem 1(d)

def calc_next_state(pe, s, a):
    """
    Calculate a next state
    :param pe:error probability
    :param s:state
    :param a:action
    :return:next state that follows the pdf

    # Idea: using random values, I calculate the next state

    """
    # s = list(a)
    P_trans = {}
    Prob_tmp = {}
    next_state = {}
    P_trans[0] = [([0, 1], 0, 1 - 2 * pe), ([0, 1], 11, pe), ([0, 1], 1, pe)]
    P_trans[1] = [([0, 1], 1, 1 - 2 * pe), ([0, 1], 0, pe), ([1, 0], 2, pe)]
    P_trans[2] = [([1, 0], 2, 1 - 2 * pe), ([0, 1], 1, pe), ([1, 0], 3, pe)]
    P_trans[3] = [([1, 0], 3, 1 - 2 * pe), ([1, 0], 2, pe), ([1, 0], 4, pe)]
    P_trans[4] = [([1, 0], 4, 1 - 2 * pe), ([1, 0], 3, pe), ([0, -1], 5, pe)]
    P_trans[5] = [([0, -1], 5, 1 - 2 * pe), ([1, 0], 4, pe), ([0, -1], 6, pe)]
    P_trans[6] = [([0, -1], 6, 1 - 2 * pe), ([0, -1], 5, pe), ([0, -1], 7, pe)]
    P_trans[7] = [([0, -1], 7, 1 - 2 * pe), ([0, -1], 6, pe), ([-1, 0], 8, pe)]
    P_trans[8] = [([-1, 0], 8, 1 - 2 * pe), ([0, -1], 7, pe), ([-1, 0], 9, pe)]
    P_trans[9] = [([-1, 0], 9, 1 - 2 * pe), ([-1, 0], 8, pe), ([-1, 0], 10, pe)]
    P_trans[10] = [([-1, 0], 10, 1 - 2 * pe), ([-1, 0], 9, pe), ([0, 1], 11, pe)]
    P_trans[11] = [([0, 1], 11, 1 - 2 * pe), ([-1, 0], 10, pe), ([0, 1], 0, pe)]

    # Create random values that follows uniform pdf
    delta = np.random.uniform(0, 1)
    # No error case
    if delta <= 1 - 2 * pe:
        P_tmp = P_trans[s[2]][0]
    elif 1 - 2 * pe < delta and delta < 1 - pe:  # Error rotation to left
        P_tmp = P_trans[s[2]][1]
    else:  # Error rotation to right
        P_tmp = P_trans[s[2]][2]

    if a[0] == 1:  # Forward motion

        x_goal = s[0] + P_tmp[0][0]
        y_goal = s[1] + P_tmp[0][1]
        h_goal = (a[1] + P_tmp[1]) % 12
        # At edges of the grids, the robot can only rotate:
        if x_goal < 0 or x_goal > L - 1:
            x_goal = s[0]
        if y_goal < 0 or y_goal > W - 1:
            y_goal = s[1]

        next_state = [x_goal, y_goal, h_goal]

    if a[0] == -1:  # Backward motion

        x_goal = s[0] - P_tmp[0][0]
        y_goal = s[1] - P_tmp[0][1]
        h_goal = (a[1] + P_tmp[1]) % 12
        # At edges of the grids, the robot can only rotate:
        if x_goal < 0 or x_goal > L - 1:
            x_goal = s[0]
        if y_goal < 0 or y_goal > W - 1:
            y_goal = s[1]

        next_state = [x_goal, y_goal, h_goal]

    if a[0] == 0:  # No linear motion, i.e. no error
        x_goal = s[0]
        y_goal = s[1]
        h_goal = (a[1] + s[2]) % 12
        next_state = [x_goal, y_goal, h_goal]

    return next_state


# Test for Problem 1(d)
s = (0, 3, 4)
pe = 0.25
a = [1, 0]
nextS = calc_next_state(pe, s, a)
print(nextS)


# Problem 2(a)

def reward(s):
    """
    Calculate a next state
    :param pe:error probability
    :param s:state
    :param a:action
    :return:next state that follows the pdf

    # Idea: using random values, I calculate the next state

    """
    reward = 0
    if (s[0] in [0, 7]) or (s[1] in [0, 7]):
        reward = -100
    if s[0] == 3 and s[1] in [4, 5, 6]:
        reward = -10
    if s[0] == 5 and s[1] == 6:
        reward = 1

    return reward


# Test for Problem 2(a)
s = (0, 3, 4)
print(reward(s))

#### Problem 3
# Problem 3(a)
# Initialization
policy = {}
goal_xy_state = [5, 6]
goal_h_state = 0

for s in S:
    # For all state, let's calculate the length between the current location and the goal location. In addition,
    # let's calculate the orientation difference, too.
    tmp_action = [0, 0]
    diff = goal_xy_state - s[0][0:2] #Linear difference
    diff_h = goal_h_state - s[0][2] #Rotational difference

    # If a robot is at goal location
    if diff[0] == [0] and diff[1] == [0]:
        tmp_action[0] = 0
    # If a robot is heading north:
    if s[0][2] in [11, 0, 1]:
        if diff[0] >= 0 and diff[1] >= 0: # If a location of a robot is south-west from a goal
            tmp_action[0] = 1 #Forward
        if diff[0] >= 0 and diff[1] < 0: # If a location of a robot is north-west from a goal
            tmp_action[0] = -1 #Backward
        if diff[0] < 0 and diff[1] >= 0: # If a location of a robot is south-east from a goal
            tmp_action[0] = 1 #Forward
        if diff[0] < 0 and diff[1] < 0: # If a location of a robot is north-east from a goal
            tmp_action[0] = -1 #Backward
    if s[0][2] in [2, 3, 4]:
        if diff[0] >= 0 and diff[1] >= 0: # If a location of a robot is south-west from a goal
            tmp_action[0] = 1 #Forward
        if diff[0] >= 0 and diff[1] < 0: # If a location of a robot is north-west from a goal
            tmp_action[0] = 1 #Forward
        if diff[0] < 0 and diff[1] >= 0: # If a location of a robot is south-east from a goal
            tmp_action[0] = -1
        if diff[0] < 0 and diff[1] < 0: # If a location of a robot is north-east from a goal
            tmp_action[0] = -1
    if s[0][2] in [5, 6, 7]:
        if diff[0] >= 0 and diff[1] >= 0: # If a location of a robot is south-west from a goal
            tmp_action[0] = -1
        if diff[0] >= 0 and diff[1] < 0: # If a location of a robot is north-west from a goal
            tmp_action[0] = 1 #Forward
        if diff[0] < 0 and diff[1] >= 0: # If a location of a robot is south-east from a goal
            tmp_action[0] = -1
        if diff[0] < 0 and diff[1] < 0: # If a location of a robot is north-east from a goal
            tmp_action[0] = 1 #Forward
    if s[0][2] in [8, 9, 10]:
        if diff[0] >= 0 and diff[1] >= 0: # If a location of a robot is south-west from a goal
            tmp_action[0] = -1
        if diff[0] >= 0 and diff[1] < 0: # If a location of a robot is north-west from a goal
            tmp_action[0] = -1
        if diff[0] < 0 and diff[1] >= 0: # If a location of a robot is south-east from a goal
            tmp_action[0] = 1 #Forward
        if diff[0] < 0 and diff[1] < 0: # If a location of a robot is north-east from a goal
            tmp_action[0] = 1 #Forward
    # If a robot is at goal:
    if diff[0] == [0] and diff[1] == [0]:
        tmp_action[0] = 0

    # About heading state:
    temp_h = 0
    # Angle difference from the current state and the goal state:
    temp_h = math.atan2(diff[1], diff[0])
    current_h = s[0][2]
    # calculate corresponding angle
    current_rad = (90 - 30 * current_h) * math.pi / 180
    # Move the robot's posture so that the difference between the current posture and the goal posture becomes smaller.
    if (temp_h - current_rad) < math.pi and (temp_h - current_rad) > 0:
        tmp_action[1] = -1
    elif (temp_h - current_rad) == math.pi and (temp_h - current_rad) == 0:
        tmp_action[1] = 0
    else:
        tmp_action[1] = 1

    policy[tuple(s[0])] = tmp_action


# Problem 3(b)
def generate_plot_trajectory(policy, s0, pe):
    """
    Generate and plot a trajectory of a robot
    :param policy: given policy
    :param s0: initial state
    :param pe:error probability
    :return:trajectory of a robot & figure of the trajectory
    """

    trajectory = []
    s_current = s0
    count = 1
    while True:
        trajectory.append([(s_current), policy[tuple(s_current)]])
        # If a robot is at the goal
        if s_current[0] == 5 and s_current[1] == 6:
            break
        s_current = calc_next_state(pe, s_current, policy[tuple(s_current)])
        count += 1

    # Plot part
    fig = plt.figure(figsize=(L, W))
    ax = fig.add_subplot(1, 1, 1)
    plt.xlim((0, L))
    plt.ylim((0, W))
    plt.grid(True, color='k')

    # Plot boarder area
    edge1 = plt.Rectangle((0, 0), 1, W, color='r')
    edge2 = plt.Rectangle((0, 7), W, 1, color='r')
    edge3 = plt.Rectangle((7, 0), 1, W, color='r')
    edge4 = plt.Rectangle((0, 0), W, 1, color='r')
    ax.add_patch(edge1)
    ax.add_patch(edge2)
    ax.add_patch(edge3)
    ax.add_patch(edge4)
    # Plot yellow area
    yello1 = plt.Rectangle((3, 4), 1, 3, color='gold')
    ax.add_patch(yello1)
    # Plot green goal
    greengoal = plt.Rectangle((5, 6), 1, 1, color='lime')
    ax.add_patch(greengoal)

    plt.plot(s0[0] + 0.5, s0[1] + 0.5, 'o', markersize='10.5', color='k')
    ax.arrow(s0[0] + 0.5, s0[1] + 0.5, 0.4 * np.sin(30 * s0[2] * np.pi / 180), 0.4 * np.cos(30 * s0[2] * np.pi / 180), \
             head_width=0.1, head_length=0.2, fc='k', ec='k')

    for i in range(1, len(trajectory)):
        xplot = trajectory[i - 1][0][0]
        yplot = trajectory[i - 1][0][1]
        nextxplot = trajectory[i][0][0]
        nextyplot = trajectory[i][0][1]
        nexthplot = trajectory[i][0][2]
        plt.plot([xplot + 0.5, nextxplot + 0.5], [yplot + 0.5, nextyplot + 0.5], 'k--')
        plt.plot(nextxplot + 0.5, nextyplot + 0.5, 'o', markersize='10.5', color='k')
        ax.arrow(nextxplot + 0.5, nextyplot + 0.5, 0.4 * np.sin(30 * nexthplot * np.pi / 180),
                 0.4 * np.cos(30 * nexthplot * np.pi / 180), \
                 head_width=0.1, head_length=0.2, fc='k', ec='k')
        plt.xlabel('x')
        plt.ylabel('y')
        # plt.bar(align="center")

    plt.show()

    return trajectory





# Problem 3(c)
tra = generate_plot_trajectory(policy, [1, 6, 6], 0)
print('Initial policy is as follows: ',tra)


# Problem 3(d)
def policy_evaluation(policy, discount_factor, pe, threshold = 0.0001):
    """
    Evaluate policy
    :param policy: given policy
    :param discount_factor: discount_factor
    :param pe: error possibility
    :param threshold: this is used to stop the iteration
    :return:trajectory of a robot & figure of the trajectory
    """
    V = {} # dictionary for value function
    listS= list(S)
    s_current = []

    # Initialize
    for s in listS:
        V[tuple(s[0])] = 0
    while True:
        delta_test = 0
        for s in listS:
            v = 0
            tmps = s.tolist()
            s_current = tmps[0]


            a = policy[tuple(s_current)]
            s = s_current

            P_trans = {}
            next_state = {}
            P_trans[0] = [([0, 1], 0, 1 - 2 * pe), ([0, 1], 11, pe), ([0, 1], 1, pe)]
            P_trans[1] = [([0, 1], 1, 1 - 2 * pe), ([0, 1], 0, pe), ([1, 0], 2, pe)]
            P_trans[2] = [([1, 0], 2, 1 - 2 * pe), ([0, 1], 1, pe), ([1, 0], 3, pe)]
            P_trans[3] = [([1, 0], 3, 1 - 2 * pe), ([1, 0], 2, pe), ([1, 0], 4, pe)]
            P_trans[4] = [([1, 0], 4, 1 - 2 * pe), ([1, 0], 3, pe), ([0, -1], 5, pe)]
            P_trans[5] = [([0, -1], 5, 1 - 2 * pe), ([1, 0], 4, pe), ([0, -1], 6, pe)]
            P_trans[6] = [([0, -1], 6, 1 - 2 * pe), ([0, -1], 5, pe), ([0, -1], 7, pe)]
            P_trans[7] = [([0, -1], 7, 1 - 2 * pe), ([0, -1], 6, pe), ([-1, 0], 8, pe)]
            P_trans[8] = [([-1, 0], 8, 1 - 2 * pe), ([0, -1], 7, pe), ([-1, 0], 9, pe)]
            P_trans[9] = [([-1, 0], 9, 1 - 2 * pe), ([-1, 0], 8, pe), ([-1, 0], 10, pe)]
            P_trans[10] = [([-1, 0], 10, 1 - 2 * pe), ([-1, 0], 9, pe), ([0, 1], 11, pe)]
            P_trans[11] = [([0, 1], 11, 1 - 2 * pe), ([-1, 0], 10, pe), ([0, 1], 0, pe)]


            if a[0] == 1:  # Forward motion

                for result in P_trans[s[2]]:
                    x_goal = s[0] + result[0][0]
                    y_goal = s[1] + result[0][1]
                    h_goal = (a[1] + result[1]) % 12
                # At edges of the grids, the robot can only rotate:
                    if x_goal < 0 or x_goal > L - 1:
                        x_goal = s[0]
                    if y_goal < 0 or y_goal > W - 1:
                        y_goal = s[1]

                    if result[2] != 0.0:
                        next_state[result[2]] = (x_goal, y_goal, h_goal)

            if a[0] == -1:  # Backward motion

                for result in P_trans[s[2]]:
                    x_goal = s[0] - result[0][0]
                    y_goal = s[1] - result[0][1]
                    h_goal = (a[1] + result[1]) % 12
                    # At edges of the grids, the robot can only rotate:
                    if x_goal < 0 or x_goal > L - 1:
                        x_goal = s[0]
                    if y_goal < 0 or y_goal > W - 1:
                        y_goal = s[1]

                    if result[2] != 0.0:
                        next_state[result[2]] = (x_goal, y_goal, h_goal)

            if a[0] == 0:  # No linear motion, i.e. no error
                x_goal = s[0]
                y_goal = s[1]
                h_goal = (a[1] + s[2]) % 12
                next_state[1.0] = (x_goal, y_goal, h_goal)



            # Search all potential next step. Then, caclulate value.
            for probability in next_state.keys():
                nextS = next_state[probability]
                v = v + probability * (reward(tuple(nextS))+discount_factor*V[tuple(nextS)])

            delta_test = max(delta_test, np.abs(v-V[tuple(s)]))
            V[tuple(s)] = v
            # delta_test = np.abs(v-V[tuple(s)])
        # Stop iteration
        if delta_test < threshold:
            break
    return V





# Problem 3(e)
V = policy_evaluation(policy,discount_factor=0.9,pe=0.0)
for tmp in range(len(tra)):
    print("Value along trajectory",tra[tmp][0],V[tuple(tra[tmp][0])])


# Problem 3(f)
def one_step_lookahead(V,pe=0.0,discount_factor = 0.9):
    """
    Compute an optimal policy given one-step lookhead on V
    :param V: value function for current policy
    :param pe: error possibility
    :param discount_factor: discount factor for future reward
    :return: optimal policy
    """
    policy = {}
    tmp_action = {}


    for tmps in S:
        action = np.zeros(nA)
        for i in range(len(A[0])):
            tmp_action[i] = [A[0][i],A[1][i]]



            a = tuple(tmp_action[i])
            s = tuple(tmps[0])

            P_trans = {}
            next_state = {}
            P_trans[0] = [([0, 1], 0, 1 - 2 * pe), ([0, 1], 11, pe), ([0, 1], 1, pe)]
            P_trans[1] = [([0, 1], 1, 1 - 2 * pe), ([0, 1], 0, pe), ([1, 0], 2, pe)]
            P_trans[2] = [([1, 0], 2, 1 - 2 * pe), ([0, 1], 1, pe), ([1, 0], 3, pe)]
            P_trans[3] = [([1, 0], 3, 1 - 2 * pe), ([1, 0], 2, pe), ([1, 0], 4, pe)]
            P_trans[4] = [([1, 0], 4, 1 - 2 * pe), ([1, 0], 3, pe), ([0, -1], 5, pe)]
            P_trans[5] = [([0, -1], 5, 1 - 2 * pe), ([1, 0], 4, pe), ([0, -1], 6, pe)]
            P_trans[6] = [([0, -1], 6, 1 - 2 * pe), ([0, -1], 5, pe), ([0, -1], 7, pe)]
            P_trans[7] = [([0, -1], 7, 1 - 2 * pe), ([0, -1], 6, pe), ([-1, 0], 8, pe)]
            P_trans[8] = [([-1, 0], 8, 1 - 2 * pe), ([0, -1], 7, pe), ([-1, 0], 9, pe)]
            P_trans[9] = [([-1, 0], 9, 1 - 2 * pe), ([-1, 0], 8, pe), ([-1, 0], 10, pe)]
            P_trans[10] = [([-1, 0], 10, 1 - 2 * pe), ([-1, 0], 9, pe), ([0, 1], 11, pe)]
            P_trans[11] = [([0, 1], 11, 1 - 2 * pe), ([-1, 0], 10, pe), ([0, 1], 0, pe)]


            if a[0] == 1:  # Forward motion
                # For every possible action:
                for result in P_trans[s[2]]:
                    x_goal = s[0] + result[0][0]
                    y_goal = s[1] + result[0][1]
                    h_goal = (a[1] + result[1]) % 12
                # At edges of the grids, the robot can only rotate:
                    if x_goal < 0 or x_goal > L - 1:
                        x_goal = s[0]
                    if y_goal < 0 or y_goal > W - 1:
                        y_goal = s[1]
                    # Add the possible next state if the possibility of achiving there is not 0 %
                    if result[2] != 0.0:
                        next_state[result[2]] = (x_goal, y_goal, h_goal)

            if a[0] == -1:  # Backward motion

                for result in P_trans[s[2]]:
                    x_goal = s[0] - result[0][0]
                    y_goal = s[1] - result[0][1]
                    h_goal = (a[1] + result[1]) % 12
                    # At edges of the grids, the robot can only rotate:
                    if x_goal < 0 or x_goal > L - 1:
                        x_goal = s[0]
                    if y_goal < 0 or y_goal > W - 1:
                        y_goal = s[1]

                    if result[2] != 0.0:
                        next_state[result[2]] = (x_goal, y_goal, h_goal)

            if a[0] == 0:  # No linear motion, i.e. no error
                x_goal = s[0]
                y_goal = s[1]
                h_goal = (a[1] + s[2]) % 12
                next_state[1.0] = (x_goal, y_goal, h_goal)#Possibility is always 1 (100 %)

            # Search all next state
            for probability in next_state.keys():
                nextS = next_state[probability]
                # Calculate Q value
                action[i] += probability*(reward(tuple(nextS))+discount_factor*V[tuple(nextS)])
        # Find the optimal Q.
        optimal_num = np.argmax(action)
        policy[tuple(s)] = [A[0][optimal_num],A[1][optimal_num]]

    return policy

one_step_lookahead(V)
print(policy)

# Problem 3(g)
def policy_iteration(policy, p_e=0.0, discount_factor=0.9):
    """
    Policy Iteration


    policy: initial policy used to iterate
    p_e: the error probability
    discount_factor: discount factor.

    Returns:
        optimal policy and optimal value
    """
    while True:
        V = policy_evaluation(policy,discount_factor, p_e)

        # To judge whether the policy converges or not:
        frag = True

        new_policy = one_step_lookahead(V, p_e)

        if policy != new_policy:
            frag = False
        policy = new_policy

        if frag:
            return [policy, V]



# Problem 3(h) and 3(i)


start_time = time.time()
optimal_p, optimal_v = policy_iteration(policy)
end_time = time.time()
print("Time of computation using policy iteration is: %s s" %str(end_time - start_time) )
s0 = (1,6,6)
traj = generate_plot_trajectory(optimal_p,s0,pe=0.0)
print('Optimal policy is as follows: ',traj)
for tmp in range(len(traj)):
    print("Value along trajectory",traj[tmp][0],optimal_v[tuple(traj[tmp][0])])



# Problem 4(a)
def value_iteration(policy, pe=0.0, discount_factor=0.9, threshold = 0.0001):
    """
    Value Iteration


    policy: initial policy
    pe: the error probability
    discount_factor: discount factor
    threshold: threshold for iteration

    Returns:
        optimal policy and optimal value
    """
    # Create dictionary for value functions
    V = {}
    tmp_action = {}
    # Initialize
    for s in S:
        V[tuple(s[0])] = 0
    # Iteration
    while True:
        delta_test = 0
        for tmps in S:
            action = np.zeros(nA)

            for i in range(len(A[0])):
                tmp_action[i] = [A[0][i], A[1][i]]

                a = tuple(tmp_action[i])
                s = tuple(tmps[0])

                P_trans = {}
                next_state = {}
                P_trans[0] = [([0, 1], 0, 1 - 2 * pe), ([0, 1], 11, pe), ([0, 1], 1, pe)]
                P_trans[1] = [([0, 1], 1, 1 - 2 * pe), ([0, 1], 0, pe), ([1, 0], 2, pe)]
                P_trans[2] = [([1, 0], 2, 1 - 2 * pe), ([0, 1], 1, pe), ([1, 0], 3, pe)]
                P_trans[3] = [([1, 0], 3, 1 - 2 * pe), ([1, 0], 2, pe), ([1, 0], 4, pe)]
                P_trans[4] = [([1, 0], 4, 1 - 2 * pe), ([1, 0], 3, pe), ([0, -1], 5, pe)]
                P_trans[5] = [([0, -1], 5, 1 - 2 * pe), ([1, 0], 4, pe), ([0, -1], 6, pe)]
                P_trans[6] = [([0, -1], 6, 1 - 2 * pe), ([0, -1], 5, pe), ([0, -1], 7, pe)]
                P_trans[7] = [([0, -1], 7, 1 - 2 * pe), ([0, -1], 6, pe), ([-1, 0], 8, pe)]
                P_trans[8] = [([-1, 0], 8, 1 - 2 * pe), ([0, -1], 7, pe), ([-1, 0], 9, pe)]
                P_trans[9] = [([-1, 0], 9, 1 - 2 * pe), ([-1, 0], 8, pe), ([-1, 0], 10, pe)]
                P_trans[10] = [([-1, 0], 10, 1 - 2 * pe), ([-1, 0], 9, pe), ([0, 1], 11, pe)]
                P_trans[11] = [([0, 1], 11, 1 - 2 * pe), ([-1, 0], 10, pe), ([0, 1], 0, pe)]

                if a[0] == 1:  # Forward motion

                    for result in P_trans[s[2]]:
                        x_goal = s[0] + result[0][0]
                        y_goal = s[1] + result[0][1]
                        h_goal = (a[1] + result[1]) % 12
                        # At edges of the grids, the robot can only rotate:
                        if x_goal < 0 or x_goal > L - 1:
                            x_goal = s[0]
                        if y_goal < 0 or y_goal > W - 1:
                            y_goal = s[1]

                        if result[2] != 0.0:
                            next_state[result[2]] = (x_goal, y_goal, h_goal)

                if a[0] == -1:  # Backward motion

                    for result in P_trans[s[2]]:
                        x_goal = s[0] - result[0][0]
                        y_goal = s[1] - result[0][1]
                        h_goal = (a[1] + result[1]) % 12
                        # At edges of the grids, the robot can only rotate:
                        if x_goal < 0 or x_goal > L - 1:
                            x_goal = s[0]
                        if y_goal < 0 or y_goal > W - 1:
                            y_goal = s[1]

                        if result[2] != 0.0:
                            next_state[result[2]] = (x_goal, y_goal, h_goal)

                if a[0] == 0:  # No linear motion, i.e. no error
                    x_goal = s[0]
                    y_goal = s[1]
                    h_goal = (a[1] + s[2]) % 12
                    next_state[1.0] = (x_goal, y_goal, h_goal)

                for probability in next_state.keys():
                    nextS = next_state[probability]
                    action[i] += probability * (reward(tuple(nextS)) + discount_factor * V[tuple(nextS)])


            optimal_action = np.max(action)
            optimal_index = np.argmax(action)
            policy[tuple(s)] = [A[0][optimal_index], A[1][optimal_index]]
            # To judge whether iteration ends or not
            delta_test = max(delta_test, np.abs(optimal_action - V[tuple(s)]))

            V[tuple(s)] = optimal_action
        # If difference between current values and one step before value is smaller than threshold, stop iteration
        if delta_test < threshold:
            break
    return policy, V

# Problem 4(b) and (c)
start_time = time.time()
optimal_p, optimal_v = value_iteration(policy, pe =0.0, discount_factor=0.9)
end_time = time.time()
print("Time of computation using value iteration is: %s s" %str(end_time - start_time) )

s0 = (1, 6, 6)
traj = generate_plot_trajectory(optimal_p,s0,pe=0.0)
print('Optimal policy is as follows: ',traj)
for tmp in range(len(traj)):
    print("Value along trajectory",traj[tmp][0],optimal_v[tuple(traj[tmp][0])])





# Problem 5(a)
optimal_p, optimal_v = policy_iteration(policy, p_e=0.25)
s0 = (1,6,6)
traj = generate_plot_trajectory(optimal_p,s0,pe=0.25)
print('Optimal policy is as follows: ',traj)
for tmp in range(len(traj)):
    print("Value along trajectory",traj[tmp][0],optimal_v[tuple(traj[tmp][0])])
optimal_p, optimal_v = value_iteration(policy, pe=0.25)
s0 = (1,6,6)
traj = generate_plot_trajectory(optimal_p,s0,pe=0.25)
print('Optimal policy is as follows: ',traj)
for tmp in range(len(traj)):
    print("Value along trajectory",traj[tmp][0],optimal_v[tuple(traj[tmp][0])])