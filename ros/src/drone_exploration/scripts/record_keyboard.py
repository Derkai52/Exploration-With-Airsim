from gym_env.envs.drone_explore_env import DroneExploreEnv
import numpy as np
import time

#env1 = DroneExploreEnv(drone_name = 'Drone_1')
env2 = DroneExploreEnv(drone_name = 'Drone_2')
#env1.render("0")
env2.render("0")

save_path = "/media/${USER}/Data/Graduation_Data/Teacher_Data/"
start_time1 = time.time()
actions = []
rewards = []
observations = []

observations.append(env2.reset())

i=0
while(i<300):
    action = env2.keboard_to_action_8()
    actions.append(action)
    obs,reward,_,_ = env2.step(action)
    rewards.append(reward)
    observations.append(obs)
    i += 1

actions_path = save_path+'actions/'+str(start_time1)+'.npy'
rewards_path = save_path+'rewards/'+str(start_time1)+'.npy'
observations_path = save_path+'observations/'+str(start_time1)+'.npy'
np.save(actions_path,np.array(actions,dtype=int))
np.save(rewards_path,np.array(rewards,dtype=float))
np.save(observations_path,np.array(observations,dtype=float))
final_time1 =  time.time()
print(start_time1)
print(final_time1)
print(start_time1 - final_time1)

# a = np.array([[0,1,2,3,2,3],[0,1,2,3,0,1]])
# print(a)
# b = np.where(a != 1)
# print(len(b[0]))

