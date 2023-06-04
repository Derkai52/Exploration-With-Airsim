from gym_env.envs.drone_explore_env import DroneExploreEnv
import numpy as np
import time

env1 = DroneExploreEnv(drone_name = 'Drone_1')
env2 = DroneExploreEnv(drone_name = 'Drone_2')
env1.render("0")
env2.render("0")

start_time1 = time.time()


i=0
while(i<5000):
    action = env1.keboard_to_action_8()
    if i%2 == 0:
        env1.step(action)
    else:
        env2.step(action)
    i += 1
final_time1 =  time.time()
env1.reset()
print("Rest----1")
action = env1.keboard_to_action_8()
env2.reset()
print("Rest----2")
action = env1.keboard_to_action_8()
i=0
while(i<20):
    action = env1.keboard_to_action_8()
    if i%2 == 0:
        env1.step(action)
    else:
        env2.step(action)
    i += 1
print(start_time1)
print(final_time1)
print(start_time1 - final_time1)

# a = np.array([[0,1,2,3,2,3],[0,1,2,3,0,1]])
# print(a)
# b = np.where(a != 1)
# print(len(b[0]))

