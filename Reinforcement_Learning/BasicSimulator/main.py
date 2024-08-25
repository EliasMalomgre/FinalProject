# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import random
import math
from Environment import CarWorld

speedLimit = 70

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    env = CarWorld()

    for i in range(100):
        print("Current state: ", env.state)
        action = random.randint(0, 6)
        print("Action: ", action-3)
        env.step(action)
        print("New State: ", env.state)
        print()

def test():
    reward = 0
    index = 0

    while True:
        ownSpeed = math.sin(index / 10) * 40 + 40
        index += 1
        reward = 0
        output = random.randint(-3, 3)
        distance = random.randint(0, 50)

        print("Output: ", output, "Distance: ", distance, "Own Speed", ownSpeed)

        if ownSpeed < speedLimit:
            if output > 0:
                if distance > 4:
                    reward += 1
                    print(reward, " Good")
                elif distance < 4:
                    reward -= 1
                    print(reward, " Bad")
            elif output < 0:
                if distance < 4:
                    reward += 1
                    print(reward, " Good")
                elif distance > 4:
                    reward -= 1
                    print(reward, " Bad")
            else:  # output is 0
                if distance not in range(4, 9):
                    reward -= 1
                    print(reward, " Bad")
                else:
                    reward += 1
                    print(reward, " Good")
        else:
            if output >= 0:
                reward -= 1
                print(reward, " Bad")
            else:
                reward += 1
                print(reward, " Good")
