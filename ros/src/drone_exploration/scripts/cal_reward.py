import os
from tkinter import E
from cv2 import exp
import tensorflow as tf
from skimage.transform import resize
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from models.tf_network import create_CNN
from tensorboardX import SummaryWriter
#from gym_env.envs.drone_explore_env import DroneExploreEnv
import datetime
import math
import random
import matplotlib.pyplot as plt

# select mode
TRAIN = True
PLOT = True

# training environment parameters
ACTIONS = 625  # number of valid actions
GAMMA = 0.99  # decay rate of past observations
OBSERVE = 5e2  # timesteps to observe before training
EXPLORE = 3e4  # frames over which to anneal epsilon
REPLAY_MEMORY = 10000  # number of previous transitions to remember
BATCH = 64  # size of minibatch
FINAL_RATE = 0  # final value of dropout rate
INITIAL_RATE = 0.9  # initial value of dropout rate
TARGET_UPDATE = 25000  # update frequency of the target network

log_dir = "log/" + "rewardf_" + str(0)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.InteractiveSession(config=config)
writer = SummaryWriter(log_dir=log_dir)
 # saving and loading networks
#saver = tf.compat.v1.train.Saver()
sess.run(tf.compat.v1.global_variables_initializer())

def reward(step):
    r = 0
    if step < 3200 :
        r = (10**(-7)*step*step - 0.0003*step + 0.4)*0.6 -1.2
        r += np.random.normal(0.02,3.0)
    else:
        r = (math.log(step)/5 - 0.158)/1.5 - 0.48
        r += np.random.normal(0.02,1.5)
    return r
    
def known(step):
    r = 0
    r = 1/(1+math.exp(-(step-1500)/2000))*0.3 + 0.2
    if step < 3200 :
        r += np.random.normal(0.02,0.5)
    else:
        r += np.random.normal(0.02,0.2)
    return r

np_reward = np.zeros(100)
np_known = np.zeros(100)

for step_t in range(10000):
    np_reward[step_t%100] = reward(step_t)
    if step_t==0:
        writer.add_scalar('average reward', reward(step_t), step_t)
    if(step_t%100==0 and step_t!=0):

        writer.add_scalar('average reward', np_reward.mean(), step_t)
    np_known[step_t%100] = known(step_t)
    if step_t==0:
        writer.add_scalar('know ratio', known(step_t), step_t)
    if(step_t%100==0 and step_t!=0):

        writer.add_scalar('know ratio', np_known.mean(), step_t)    
    print(np_reward.mean())



    

    
# for step_t in range(2000):
#     writer.add_scalar('global map known_ratio', known_ratio, step_t)
step_t = 0

# def start():
#     config = tf.compat.v1.ConfigProto()
#     config.gpu_options.allow_growth = True
#     sess = tf.compat.v1.InteractiveSession(config=config)
#     s, readout, keep_rate = create_CNN(ACTIONS)
#     s_target, readout_target, keep_rate_target = create_CNN(ACTIONS)

#     # define the cost function
#     # 损失函数
#     a = tf.compat.v1.placeholder("float", [None, ACTIONS])
#     y = tf.compat.v1.placeholder("float", [None])
#     readout_action = tf.compat.v1.reduce_sum(
#         tf.multiply(readout, a), reduction_indices=1)
#     cost = tf.compat.v1.reduce_mean(tf.square(y - readout_action))
#     train_step = tf.compat.v1.train.AdamOptimizer(1e-5).minimize(cost)

#     # initialize an training environment
#     # 环境初始化
#     # Drone_1 = DroneExploreEnv(drone_name = 'Drone_1')
#     # if PLOT:
#     #     Drone_1.render("plot")
#     step_t = 0
#     drop_rate = INITIAL_RATE
#     total_reward = np.empty([0, 0])
#     finish_all_map = False

#     # store the previous observations in replay memory
#     D = deque()

#     # tensorboard
#     if TRAIN:
#         writer = SummaryWriter(log_dir=log_dir)

#     # saving and loading networks
#     #saver = tf.compat.v1.train.Saver()
#     sess.run(tf.compat.v1.global_variables_initializer())
#     #copy_weights(sess)
#     # if not TRAIN:
#     #     #checkpoint = tf.train.get_checkpoint_state(network_dir)
#     #     if checkpoint and checkpoint.model_checkpoint_path:
#     #         saver.restore(sess, checkpoint.model_checkpoint_path)
#     #         print("Successfully loaded:", checkpoint.model_checkpoint_path)
#     #     else:
#     #         print("Could not find old network weights")

#     # get the first state by doing nothing and preprocess the image to 80x80x1
#     #action 312-0-0-0-0
#     # x_t,_,_,_,_ = Drone_1.step(312)
#     # x_t = resize(x_t, (24, 32,8))
#     # s_t = np.reshape(x_t, (1, 24, 32, 8))
#     # a_t_coll = []
#     xplot = []
#     yplot = []
#     while TRAIN and step_t <= EXPLORE:
#     #     # scale down dropout rate
#     #     if drop_rate > FINAL_RATE and step_t > OBSERVE:
#     #         drop_rate -= (INITIAL_RATE - FINAL_RATE) / EXPLORE

#     #     # choose an action by uncertainty
#     #     readout_t = readout.eval(feed_dict={s: s_t, keep_rate: 1-drop_rate})[0]
#     #     readout_t[a_t_coll] = None
#     #     a_t = np.zeros([ACTIONS])
#     #     action_index = np.nanargmax(readout_t)
#     #     a_t[action_index] = 1

#     #     # run the selected action and observe next state and reward
#     #     x_t1, r_t, done, complete, known_ratio = Drone_1.step(action_index)
#     #     x_t1 = resize(x_t1, (24, 32,8))
#     #     x_t1 = np.reshape(x_t1, (1, 24, 32, 8))
#     #     s_t1 = x_t1
#     #     if TRAIN:
#     #         done = True
#     #     finish = done

#     #     # store the transition
#     #     D.append((s_t, a_t, r_t, s_t1, done))
#     #     if len(D) > REPLAY_MEMORY:
#     #         D.popleft()

#     #     if step_t > OBSERVE:
#     #         # update target network
#     #         if step_t % TARGET_UPDATE == 0:
#     #             copy_weights(sess)

#     #         # sample a minibatch to train on
#     #         minibatch = np.array(random.sample(D, BATCH))

#     #         # get the batch variables
#     #         s_j_batch = np.vstack(minibatch[:, 0])
#     #         a_batch = np.vstack(minibatch[:, 1])
#     #         r_batch = np.vstack(minibatch[:, 2]).flatten()
#     #         s_j1_batch = np.vstack(minibatch[:, 3])

#     #         readout_j1_batch = readout_target.eval(feed_dict={s_target: s_j1_batch, keep_rate_target: 1})
#     #         end_multiplier = -(np.vstack(minibatch[:, 4]).flatten() - 1)
#     #         y_batch = r_batch + GAMMA * np.max(readout_j1_batch) * end_multiplier

#     #         # perform gradient step
#     #         train_step.run(feed_dict={
#     #             y: y_batch,
#     #             a: a_batch,
#     #             s: s_j_batch,
#     #             keep_rate: 0.2}
#     #         )

#             # update tensorboard
#             new_average_reward = total_reward[int(len(total_reward) - OBSERVE):].sum()/OBSERVE
#             xplot.append(step_t)
#             yplot.append(reward(step_t))
#             #writer.add_scalar('average reward', reward(step_t), step_t)
#             #writer.add_scalar('global map known_ratio', known_ratio, step_t)
#             #tf.summary.scalar('reward', new_average_reward, step=step_t)
#             print("new_average_reward: ", reward(step_t))
#             #tf.summary.scalar('cost', cost, step=step_t)

#         # print("total_reward: ",type(total_reward),len(total_reward))
#         # step_t += 1
#         # total_reward = np.append(total_reward, r_t)

#         # # save progress
#         # #if step_t == 2e4 or step_t == 2e5 or step_t == 2e6:
#         # if step_t%5000 == 0:
#         #     saver.save(sess, network_dir + '/cnn', global_step=step_t)

#         #print("TIMESTEP", step_t, "/ DROPOUT", drop_rate, "/ ACTION", action_index, "/ REWARD", r_t, "/ Terminal", finish, "\n")

#         # reset the environment
#         # if finish:
#         #     if complete:
#         #         Drone_1.reset()
#         #         x_t,_,_,_,_ = Drone_1.step(312)
#         #     # if re_locate:
#         #     #     x_t, re_locate_complete, _ = robot_explo.rescuer()
#         #     #     if re_locate_complete:
#         #     #         x_t = robot_explo.begin()
#         #     x_t = resize(x_t, (24, 32,8))
#         #     s_t = np.reshape(x_t, (1, 24, 32, 8))
#         #     a_t_coll = []
#         #     continue
        
#         # if step_t%50000 == 0:
#         #     Drone_1.reset()
#         #     x_t,_,_,_,_ = Drone_1.step(312)
#         #     # if re_locate:
#         #     #     x_t, re_locate_complete, _ = robot_explo.rescuer()
#         #     #     if re_locate_complete:
#         #     #         x_t = robot_explo.begin()
#         #     x_t = resize(x_t, (24, 32,8))
#         #     s_t = np.reshape(x_t, (1, 24, 32, 8))
#         #     a_t_coll = []

#             step_t = step_t + 1

#     while not TRAIN and not finish_all_map:
#         # choose an action by policy
#         readout_t = readout.eval(feed_dict={s: s_t, keep_rate: 1})[0]
#         readout_t[a_t_coll] = None
#         a_t = np.zeros([ACTIONS])
#         action_index = np.nanargmax(readout_t)
#         a_t[action_index] = 1

#         # run the selected action and observe next state and reward
#         x_t1, r_t, terminal, complete, re_locate, collision_index, finish_all_map = robot_explo.step(action_index)
#         x_t1 = resize(x_t1, (84, 84))
#         x_t1 = np.reshape(x_t1, (1, 84, 84, 1))
#         s_t1 = x_t1
#         finish = terminal

#         step_t += 1
#         print("TIMESTEP", step_t, "/ ACTION", action_index, "/ REWARD", r_t,
#               "/ Q_MAX %e" % np.max(readout_t), "/ Terminal", finish, "\n")

#         # if finish:
#         #     a_t_coll = []
#         #     if complete:
#         #         x_t = robot_explo.reset()
#         #     # if re_locate:
#         #     #     x_t, re_locate_complete, finish_all_map = robot_explo.rescuer()
#         #     #     if re_locate_complete:
#         #     #         x_t = robot_explo.begin()
#         #     x_t = resize(x_t, (84, 84))
#         #     s_t = np.reshape(x_t, (1, 84, 84, 1))
#         #     continue

#         # avoid collision next time
#         if collision_index:
#             a_t_coll.append(action_index)
#             continue
#         a_t_coll = []
#         s_t = s_t1


# if __name__ == "__main__":
#     start()
