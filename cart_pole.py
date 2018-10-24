import math
from random import randint
import random
import matplotlib.pyplot as plt

FORCE_MAG = 10
TOTAL_MASS = .92
GRAVITY = 9.8
LENGTH = .326
MASSPOLE = .209
POLEMASS_LENGTH = MASSPOLE * LENGTH
TAU = .02

RIGHT = 1
LEFT = 0

NUM_X_BOXES = 3
NUM_X_DOT_BOXES = 3
NUM_THETA_BOXES = 12
NUM_THETA_DOT_BOXES = 3
NUM_ACTIONS  = 2

GAMMA = .8
ALPHA = .8
EXPERIMENT = False

class SimulationState:
    def __init__(self):
        self.x = 0
        self.x_dot = 0
        self.theta = 0
        self.theta_dot = 0

    def update(self, x, x_dot, theta, theta_dot):
        self.x = x
        self.x_dot = x_dot
        self.theta = theta
        self.theta_dot = theta_dot

# action == move cart right or left
# x == position of the cart
# x_dot == velocity of cart (change of position)
# theta == angle of the pole
# theta_dot == change of pole's angle
def cart_pole(action, state):
    force = FORCE_MAG if action > 0 else -FORCE_MAG
    costheta = math.cos(state.theta)
    sintheta = math.sin(state.theta)
    temp = (force + POLEMASS_LENGTH * state.theta_dot * state.theta_dot * sintheta) / TOTAL_MASS
    thetaacc = (GRAVITY * sintheta - costheta * temp) / (LENGTH * ((4/3) - MASSPOLE * costheta * costheta / TOTAL_MASS))
    xacc = temp - POLEMASS_LENGTH * thetaacc * costheta / TOTAL_MASS
    state.x += TAU * state.x_dot
    state.x_dot += TAU * xacc
    state.theta += TAU * state.theta_dot
    state.theta_dot += TAU * thetaacc
    return state

def get_x_index(x):
    if(x < -0.8):
        index = 0
    elif(-0.8 <= x and x <= 0.8):
        index = 1
    elif(0.8 < x):
        index = 2
    return index

def get_x_dot_index(x_dot):
    if(x_dot < -0.5):
        index = 0
    elif(-0.5 <= x_dot and x_dot <= 0.5):
        index = 1
    elif(0.5 < x_dot):
        index = 2
    return index

def get_theta_index(theta):
    if(theta <= -6):
        index = 0
    elif(-6 < theta and theta <= -4):
        index = 1
    elif(-4 < theta and theta <= -3):
        index = 2
    elif(-3 < theta and theta <= -2):
        index = 3
    elif(-2 < theta and theta <= -1):
        index = 4
    elif(-1 < theta and theta <= 0):
        index = 5
    elif(0 < theta and theta <= 1):
        index = 6
    elif(1 < theta and theta <= 2):
        index = 7
    elif(2 < theta and theta <= 3):
        index = 8
    elif(3 < theta and theta <= 4):
        index = 9
    elif(4 < theta and theta <= 6):
        index = 10
    elif(6 < theta):
        index = 11
    return index

def get_theta_dot_index(theta_dot):
    if(theta_dot < -50):
        index = 0
    elif(-50 <= theta_dot and theta_dot <= 50):
        index = 1
    elif(50 < theta_dot):
        index = 2
    return index

def q_table_lookup(q, state, action):
    x_index = get_x_index(state.x)
    x_dot_index = get_x_dot_index(state.x_dot)
    theta_index = get_theta_index(state.theta)
    theta_dot_index = get_theta_dot_index(state.theta_dot)
    return q[x_index][x_dot_index][theta_index][theta_dot_index][action]

def get_possible_rewards(q, state):
    left_reward = q_table_lookup(q, state, LEFT)
    right_reward = q_table_lookup(q, state, RIGHT)
    return left_reward, right_reward

def arg_max(q, state):
    left_reward, right_reward = get_possible_rewards(q, state)
    maximized_argument = RIGHT if right_reward > left_reward else LEFT
    return maximized_argument

def q_max(q, state):
    left_reward, right_reward = get_possible_rewards(q, state)
    maximized_q_reward = right_reward if right_reward > left_reward else left_reward
    return maximized_q_reward

def get_state_reward(x_dot, theta, theta_dot):
    reward = 0
    # Reward based on theta_dot
    if ((theta < 0 and theta_dot > 0) or (theta > 0 and theta_dot < 0)):
        reward += 3
    else:
        reward -= 3
    if (EXPERIMENT == False):
        # Reward based on x_dot
        if (-1 < x_dot and x_dot < 1):
            reward += 3
        elif (-2 < x_dot and x_dot < 2):
            reward += 1
        elif (-3 < x_dot and x_dot < 3):
            reward -= 1
        elif (-4 < x_dot and x_dot < 4):
            reward -= 3
        else:
            reward -= 5
    # Reward based on theta
    if (-1 <= theta and theta <= 1):
        reward += 7
    elif (-2 <= theta and theta <= 2):
        reward += 4
    elif (-3 <= theta and theta <= 3):
        reward += 1
    elif (-4 <= theta and theta <= 4):
        reward += -3
    elif (-6 <= theta and theta <= 6):
        reward += -7
    else:
        reward += -10
    return reward

def update_q_table(q, currentState, futureState, action):
    x_index = get_x_index(currentState.x)
    x_dot_index = get_x_dot_index(currentState.x_dot)
    theta_index = get_theta_index(currentState.theta)
    theta_dot_index = get_theta_dot_index(currentState.theta_dot)
    current_q_value = q[x_index][x_dot_index][theta_index][theta_dot_index][action]
    q[x_index][x_dot_index][theta_index][theta_dot_index][action] += ALPHA * (get_state_reward(currentState.x_dot, currentState.theta, currentState.theta_dot) + GAMMA * q_max(q, futureState) - current_q_value)
    return q

def print_state(state, action):
    output = "     Action: %1i" % action
    output += "   Position: %2f" % state.x
    output += "   Velocity: %2f" % state.x_dot
    output += "   Pole Angle: %2f" % state.theta
    output += "   Change of Angle: %2f" % state.theta_dot
    print(output, "\n")

def fill_q_table(q):
    random.seed(6)
    for x in range(NUM_X_BOXES):
        q.append([])
        for x_dot in range(NUM_X_DOT_BOXES):
            q[x].append([])
            for theta in range(NUM_THETA_BOXES):
                q[x][x_dot].append([])
                for theta_dot in range(NUM_THETA_DOT_BOXES):
                    q[x][x_dot][theta].append([])
                    for action in range(NUM_ACTIONS):
                        q[x][x_dot][theta][theta_dot].append(randint(-10, 10))
    return q

def runSimulation(q):
    currentState = SimulationState()
    pastState = SimulationState()
    steps = 0
    while (42 > currentState.theta and currentState.theta > -42 and steps < 1000):
        action = arg_max(q, currentState)

        pastState.x = currentState.x
        pastState.x_dot = currentState.x_dot
        pastState.theta = currentState.theta
        pastState.theta_dot = currentState.theta_dot

        currentState = cart_pole(action, currentState)
        q = update_q_table(q, pastState, currentState, action)
        steps += 1
    return steps

def graphAttempts(controlAttempts, experimentAttempts):
    plt.plot(controlAttempts, 'green', label='Control')
    plt.plot(experimentAttempts, 'blue', label='Experiment')
    plt.title("Cart_Pole Attempts")
    plt.ylabel("Steps Balanced")
    plt.xlabel("Attempt Number")
    plt.legend()
    plt.show()

def runExperiment(numAttempts):
    q = []
    q = fill_q_table(q)
    attempt = 0
    steps = 0
    recordedAttempts = []
    while(attempt < numAttempts):
        steps = runSimulation(q)
        recordedAttempts.append(steps)
        print("Attempt: ", attempt, "     Steps: ", steps)
        attempt += 1
    return recordedAttempts


def main():
    global ALPHA
    global EXPERIMENT
    numAttempts = 100
    attemptsRecordControl = []
    attemptsRecordExperiment = []

    # ALPHA = .8
    EXPERIMENT = False
    attemptsRecordControl = runExperiment(numAttempts)

    # ALPHA = .95
    EXPERIMENT = True
    attemptsRecordExperiment = runExperiment(numAttempts)

    graphAttempts(attemptsRecordControl, attemptsRecordExperiment)

main()
