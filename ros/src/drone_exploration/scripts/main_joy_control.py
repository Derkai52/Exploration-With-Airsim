from gym_env.envs.drone_explore_env import DroneExploreEnv
import numpy as np
import time

env1 = DroneExploreEnv(drone_name = 'Drone_1')
env1.render("0")

start_time = time.time()
save_path = "/media/mrmmm/Data/Graduation_Data/Teacher_Data/"
positions = []
oritations = []
positions_path = save_path+'positions_'+str(start_time)+'.npy'
oritations_path = save_path+'oritations_'+str(start_time)+'.npy'

i=0
while(i<5000):
    action = env1.random_action()
    if i%2 == 0:
        env1.update_map()
        env1.update_viwer()
    else:
        env1.update_map()
        env1.update_viwer()
    _,position,oritation,_ = env1.get_depth_img()
    positions.append(positions)
    oritations.append(oritation)
    i += 1
    np.save(positions_path,np.array(positions))
    np.save(oritations_path,np.array(oritations))
final_time = time.time()



print(start_time)
print(final_time)
print(final_time - start_time)

# a = np.array([[0,1,2,3,2,3],[0,1,2,3,0,1]])
# print(a)
# b = np.where(a != 1)
# print(len(b[0]))

