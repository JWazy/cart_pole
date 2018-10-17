import math

FORCE_MAG = 10
TOTAL_MASS = .92
GRAVITY = 9.8
LENGTH = .326
MASSPOLE = .209
POLEMASS_LENGTH = MASSPOLE * LENGTH
TAU = .02

# action == move cart right or left
# x == position of the cart
# x_dot == velocity of cart (change of position)
# theta == angle of the pole
# theta_dot == change of pole's angle
def cart_pole(action, x, x_dot, theta, theta_dot):
    force = action > 0 if FORCE_MAG else -FORCE_MAG
    costheta = math.cos(theta)
    sintheta = math.sin(theta)
    temp = (force + POLEMASS_LENGTH * theta_dot * theta_dot * sintheta) / TOTAL_MASS
    thetaacc = (GRAVITY * sintheta - costheta * temp) / (LENGTH * ((4/3) - MASSPOLE * costheta * costheta / TOTAL_MASS))
    xacc = temp - POLEMASS_LENGTH * thetaacc * costheta / TOTAL_MASS
    x += TAU * x_dot
    x_dot += TAU * xacc
    theta += TAU * theta_dot
    theta_dot += TAU * thetaacc
    return x, x_dot, theta, theta_dot

def decide_action(x, x_dot, theta, theta_dot):
    return 1

def print_state(x, x_dot, theta, theta_dot):
    output = "Action: Right"
    output += "   Position: %2f" % x
    output += "   Velocity: %2f" % x_dot
    output += "   Pole Angle: %2f" % theta
    output += "   Change of Angle: %2f" % theta_dot
    print(output, "\n")

def main():
    x = 0
    x_dot = 0
    theta = 90
    theta_dot = 0
    while (180 > theta and theta > 0):
        action = decide_action(x, x_dot, theta, theta_dot)
        x, x_dot, theta, theta_dot = cart_pole(action, x, x_dot, theta, theta_dot)
        print_state(x, x_dot, theta, theta_dot)

main()
