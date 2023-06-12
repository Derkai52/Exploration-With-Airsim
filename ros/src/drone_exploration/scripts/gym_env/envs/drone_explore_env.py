from distutils.command.config import config
from typing import Optional, Union, Tuple, Dict, Any, List

import numpy as np
import yaml
#import octomap
#from PythonClient.airsim.types import CollisionInfo
from gym_env.envs.map_wrapper import GridMap
import matplotlib
from matplotlib import pyplot as plt
import time
import random

import gym
import gym.logger as logger
from gym import error, spaces
from gym import utils

import airsim


class DroneExploreEnv(gym.Env):
#class DroneExploreEnv():
    def __init__(
        self, 
        drone_name: str = 'Drone_1',
        client_port: float = 41451,
        config_path: str = '/home/${USER}/Exploration_With_Airsim/ros/src/drone_exploration/scripts/gym_env/configs/env_config_simple_resnet.yaml'
        ) -> None:
        super().__init__()
        #ActionSpace Mapping: https://www.zhihu.com/question/37189447/answer/74759345
        # STEP 1 读取配置文件，完成初始化
        self.drone_name = drone_name
        self.client_port = client_port
        self.config_path = config_path
        yaml_instream = open(config_path,'r',encoding='utf-8')
        yaml_config = yaml_instream.read()
        config = yaml.safe_load(yaml_config)
        self.time_ratio = config['env']['time_ratio']
        self.fps = config['env']['fps']
        self.frame_time = 1.0/self.fps #s
        self.step_mode = config['env']['step_mode']
        self.sensor_mode = config['env']['sensor_mode']
        self.action_space_len = config['env']['action_space_len']
        self.max_update_per_step = config['env']['max_update_per_step']
        self.train_mode = config['env']['train_mode']
        self.action_mode = config['drone']['action_mode']
        self.linear_ratio = config['drone']['linear_ratio']
        self.yaw_ratio = config['drone']['yaw_ratio']
        self.camera_pram = config['drone']['camera_pram']
        self.global_map_range = np.array(config['map']['global_map_range'])
        self.loacl_map_range = np.array(config['map']['loacl_map_range'])
        self.resolution = config['map']['resolution']
        self.free_ratio = config['map']['free_ratio']
        self.init_map_path = config['map']['init_map_path']
        yaml_instream.close()
        self.loacl_map_shape = np.array(self.loacl_map_range/self.resolution, dtype=np.int)

        self.map_update_grid_max = 0
        self.keyboard_action_dict = {'w': 0, 's': 1, 'a': 2,'d': 3, 'f': 4, 'r': 5,'q': 6, 'e': 7}
        self.known_region = 0
        self.rend = False

        # STEP 2 初始化Airsim,建立连接和控制，并控制起飞无人机
        self._drone_client = airsim.MultirotorClient(ip = "127.0.0.1", port = self.client_port)
        self._drone_client.confirmConnection()
        self._drone_client.enableApiControl(True, drone_name)
        self._drone_client.armDisarm(True, drone_name)
        self._drone_client.takeoffAsync(timeout_sec=20.0, vehicle_name = drone_name).join()
        #self._drone_client.moveToPositionAsync(0,0,2.0, 2.0, timeout_sec = 1.0)
        #self._drone_client.moveByVelocityAsync(0,0,2.0,10*self.frame_time,vehicle_name=self.drone_name).join()
        print("take off")

        # STEP 3 初始化地图
        # local/global_last/global_last
        self.grid_map = GridMap(self.drone_name,config_path)
        self.episodic_return = 0
        self.known_ratio = 0
        self.last_time_stamp = np.uint64(0)
        self._init_map()
        print("Init_Map")

        # STEP 4 Gym Var
        self.episodic_length = 0
        if self.action_space_len == 625:
            self._action_set = np.genfromtxt('/home/${USER}/Exploration_With_Airsim/ros/src/drone_exploration/scripts/gym_env/configs/action_space_625.csv', delimiter=",")
        elif self.action_space_len == 8:
            self._action_set = np.genfromtxt('/home/${USER}/Exploration_With_Airsim/ros/src/drone_exploration/scripts/gym_env/configs/action_space_8.csv', delimiter=",")
        self.action_space = spaces.Discrete(len(self._action_set))
        #obs_space: 0-free, 1-unknow, 2-previous_path, 3-occupy
        self.observation_space = spaces.Box(
                low=0, high=3, dtype=np.uint8, shape=self.grid_map.loacl_map_shape
            )
        self.action_index = 0

    def _init_map(self):
        if self.init_map_path == 'None':
            self._drone_client.rotateByYawRateAsync(yaw_rate = 360/50.0, duration = 62.5, vehicle_name = self.drone_name) #yaw_rate in radian per second
            self.last_time_stamp = self.get_collision_state().time_stamp
            for _ in range(36*int(self.fps)):
                if self.sensor_mode == 'depth':
                    depth,position,quaternionr,cur_time_stamp = self.get_depth_img()
                    self.grid_map.update_map_depth(depth,position,quaternionr)
                else:
                    points,position,quaternionr,cur_time_stamp = self.get_lidar_data()
                    #print(type(position),position)
                    self.grid_map.update_map_pcl(points, position, quaternionr)
                if(cur_time_stamp-self.last_time_stamp < self.frame_time * 10**9):
                    time.sleep((self.frame_time -(cur_time_stamp-self.last_time_stamp)/10**9))
                else:
                    print("超时！！！ 处理时间： ", (cur_time_stamp-self.last_time_stamp)/10**6, 'ms')
                self.last_time_stamp = cur_time_stamp
            init_map_path = '/home/${USER}/Exploration_With_Airsim/ros/src/drone_exploration/scripts/gym_env/configs/globale_map_init' + self.drone_name +'.npy'
            np.save(init_map_path,self.grid_map.global_map)
            self.init_map_path = init_map_path
        else:
            self.init_map_path = self.init_map_path + self.drone_name +'.npy'
            self._drone_client.rotateByYawRateAsync(yaw_rate = -360/6.0, duration = 1.5, vehicle_name = self.drone_name)
            GroundTruthKinematics = self._drone_client.simGetGroundTruthKinematics(vehicle_name=self.drone_name)
            self.grid_map.init_with_file(self.init_map_path, GroundTruthKinematics.position.to_numpy_array())
        self.known_region = len(np.where(self.grid_map.global_map != 1)[0])

    def get_collision_state(self):
        return self._drone_client.simGetCollisionInfo(vehicle_name = self.drone_name)

    def get_depth_img(self):
        img_response = self._drone_client.simGetImages([airsim.ImageRequest("front_center_custom", airsim.ImageType.DepthPerspective,
                                                    True, False)],vehicle_name = self.drone_name)[0]
        img1d = np.array(img_response.image_data_float, dtype=np.float)
        img2d = np.reshape(img1d, (img_response.height, img_response.width))
        return img2d, img_response.camera_position.to_numpy_array(), img_response.camera_orientation, img_response.time_stamp  

    def get_lidar_data(self):
        lidar_date = self._drone_client.getLidarData(lidar_name='lidar_1',vehicle_name=self.drone_name)
        points = np.array(lidar_date.point_cloud, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0]/3), 3))
        return points,lidar_date.pose.position.to_numpy_array(),lidar_date.pose.orientation,lidar_date.time_stamp

    def update_map(self):
        start = time.time()
        GroundTruthKinematics = self._drone_client.simGetGroundTruthKinematics(vehicle_name=self.drone_name)
        if self.sensor_mode == "depth":
            depth,position,quaternionr,self.last_time_stamp = self.get_depth_img()
            self.grid_map.update_map_depth(depth,position,quaternionr)
            #self.grid_map.update_map_depth(depth,GroundTruthKinematics.position.to_numpy_array(),GroundTruthKinematics.orientation)
        else:
            points,position,quaternionr,self.last_time_stamp = self.get_lidar_data()
            self.grid_map.update_map_pcl(points, position, quaternionr)
            #self.grid_map.update_map_pcl(points, GroundTruthKinematics.position.to_numpy_array(), GroundTruthKinematics.orientation)
        print("处理时间： ", (time.time()-start)*1000, 'ms')
        print("传感器位姿: ", position,quaternionr.to_numpy_array())
        print("GroundTruthKinematics位姿: ", GroundTruthKinematics.position.to_numpy_array(),GroundTruthKinematics.orientation.to_numpy_array())
        
    def _action_to_energy(self):
        return 0.25

    def get_info(self):
        info = {"known_ration":self.known_ratio,"known_region":self.known_region, 
                "episodic_length": self.episodic_length, "episodic_return":self.episodic_return}
        return info

    def rescure(self):
        #TODO choose a better place
        collisionInfo = self.get_collision_state()
        print("Collision!!!\n",collisionInfo)
        print("Start Rescure!!!")
        normal_position = collisionInfo.normal
        self._drone_client.moveToPositionAsync(normal_position.x_val,normal_position.y_val,normal_position.z_val, 1.0,timeout_sec=20.0, vehicle_name=self.drone_name).join()
        if self.get_collision_state().has_collided:
            print("Rescure Faild!")
        else:
            print("Rescure Succed!")

    def rescure_opp(self):
        #TODO choose a better place
        collisionInfo = self.get_collision_state()
        print("Collision!!!\n",collisionInfo)
        print("Start Rescure!!!")
        collision_position = collisionInfo.impact_point.to_numpy_array()
        GroundTruthKinematics = self._drone_client.simGetGroundTruthKinematics(vehicle_name=self.drone_name)
        self_position = GroundTruthKinematics.position.to_numpy_array()
        dst_position = self_position*2 - collision_position
        dst_position = (dst_position*2 - self_position).astype(float)
        dst_vector = airsim.Vector3r(dst_position[0], dst_position[1], dst_position[2])
        self._drone_client.moveToPositionAsync(dst_vector.x_val,dst_vector.y_val,dst_vector.z_val, 1.0,timeout_sec=20.0,vehicle_name=self.drone_name).join()
        if self.get_collision_state().has_collided:
            print("Rescure Faild!")
        else:
            print("Rescure Succed!")

    def random_action(self):
        return random.randint(0,len(self._action_set)-1)

    def action_to_control(self, action_index):
        # TODO Add jerk/acc/position mode
        self.cur_action = self._action_set[action_index]
        # self.cur_action[1] = -self.cur_action[1]
        # self.cur_action[2] = -self.cur_action[2]
        self.cur_action = self.cur_action*self.linear_ratio
        self.cur_action[3] = self.cur_action[3]/self.linear_ratio*self.yaw_ratio
        print("action_to_control: ", self.cur_action)
    
    def keboard_to_action_8(self):
        print("请输入一个动作：")
        key = input()
        try:
            self.action_index = self.keyboard_action_dict[key[0]]
        except:
            print("输入错误")
        return self.action_index
    
    def step_synchronous(self, action_ind: int):
        self.update_map()
        self.action_to_control(action_ind)
        #self.cur_action[2] = self.cur_action[2]
        if self.action_mode == 'velocity':
            self._drone_client.moveByVelocityAsync(self.cur_action[0],
                                                    -self.cur_action[1],
                                                    -self.cur_action[2],
                                                    1.6*self.frame_time,
                                                    airsim.DrivetrainType.MaxDegreeOfFreedom,
                                                    yaw_mode = airsim.YawMode(is_rate = True, yaw_or_rate = self.cur_action[3]),
                                                    vehicle_name=self.drone_name)

    def step_asynchronous(self, action_ind: int):
        self._drone_client.simPause(False)
        self.action_to_control(action_ind)
        if self.action_mode == 'velocity':
            self._drone_client.moveByVelocityAsync(self.cur_action[0],
                                                    -self.cur_action[1],
                                                    -self.cur_action[2],
                                                    self.frame_time,
                                                    airsim.DrivetrainType.MaxDegreeOfFreedom,
                                                    yaw_mode = airsim.YawMode(is_rate = True, yaw_or_rate = self.cur_action[3]),
                                                    vehicle_name=self.drone_name).join()
        #self._drone_client.moveByAngleRatesThrottleAsync(0.0, 0.0, 0.0, 0.5, self.frame_time/5, vehicle_name=self.drone_name)
        #time.sleep(self.frame_time/100)
        self._drone_client.simPause(True)
        print("Is Pause: ",self._drone_client.simIsPause())
        self.update_map()


    def seed(self, seed: Optional[int] = None):
        pass
    
    def step(self, action_ind: int):
        print(self.drone_name)
        #STEP1 执行动作，并更新地图
        done = False
        self.episodic_length += 1
        if self.step_mode == 'asynchronous':
            self.step_asynchronous(action_ind)
        else:
            self.step_synchronous(action_ind)        
        #STEP2 碰撞检测
        collisionInfo = self.get_collision_state()
        GroundTruthKinematics = self._drone_client.simGetGroundTruthKinematics(vehicle_name=self.drone_name)
        self_position = GroundTruthKinematics.position.to_numpy_array()
        if self_position[2]>0.66:
            reward = -1
            self._drone_client.takeoffAsync(timeout_sec=5.0, vehicle_name = self.drone_name).join()
            print("take off!")
            print("位姿: ", self_position,GroundTruthKinematics.orientation.to_numpy_array())
        elif collisionInfo.has_collided:
            #STEP3 碰撞时，reward=-1, 根据配置决定是否采取救援
            reward = -1
            if self.train_mode == 'rescure':
                #self.rescure()
                self.rescure_opp()
            #info = 'Collision!!!'
        else:
            #STEP3 正常情况，根据map_update_grid和energy_loss计算reward
            # Calculate map_update_rate and energy loss
            known_region = len(np.where(self.grid_map.global_map != 1)[0])
            self.known_ratio = known_region/(np.prod(self.grid_map.global_map_shape)*self.free_ratio)
            map_update_grid = known_region - self.known_region
            self.known_region = known_region
            if(self.map_update_grid_max<map_update_grid):
                self.map_update_grid_max = map_update_grid
            energy_loss = 0.25
            reward = map_update_grid/self.max_update_per_step - energy_loss
            if reward>1:
                reward = 1
            if self.known_ratio>0.8:
                done = True
            #info = "map_update_grid: " + str(map_update_grid)
            print("self.map_update_grid_max: ", self.map_update_grid_max)
            print("map_update_grid: ", map_update_grid)
            print("Known Ratio: ", self.known_ratio)
        print("reward:",reward)
        self.episodic_return += reward
        #STEP4 可视化
        if self.rend and self.episodic_length%10==0:
            self.update_viwer()
        return np.swapaxes(self.grid_map.local_map,0,2)/3, reward, done, self.get_info()
        
    def reset(self, num=0):
        print("Reset")
        # collisionInfo = self.get_collision_state()
        # if collisionInfo.has_collided:
        #     #self.rescure()
        #     self.rescure_opp()
        #self._drone_client.goHomeAsync(timeout_sec=10.0,vehicle_name=self.drone_name).join()
        if num%2==0:
            self._drone_client.reset()
        self._drone_client.confirmConnection()
        self._drone_client.enableApiControl(True, self.drone_name)
        self._drone_client.armDisarm(True, self.drone_name)
        self._drone_client.takeoffAsync(timeout_sec=20.0, vehicle_name = self.drone_name).join()
        self.episodic_length = 0
        self.episodic_return = 0
        # local/global_last/global_last
        GroundTruthKinematics = self._drone_client.simGetGroundTruthKinematics(vehicle_name=self.drone_name)
        self.grid_map.init_with_file(self.init_map_path, GroundTruthKinematics.position.to_numpy_array())
        self._drone_client.rotateByYawRateAsync(yaw_rate = -360/6.0, duration = 1.5, vehicle_name = self.drone_name).join()
        self.known_region = len(np.where(self.grid_map.global_map != 1)[0])
        self.known_ratio = self.known_region/(np.prod(self.grid_map.global_map_shape)*self.free_ratio)
        self.last_time_stamp = np.uint64(0)
        return np.swapaxes(self.grid_map.local_map,0,2)/3

    def sim_pause(self):
        self._drone_client.simPause(True)

    def sim_start(self):
        self._drone_client.simPause(False)

    def close(self) -> None:
        """
        Cleanup any leftovers by the environment
        """
        pass

    def render(self, mode: str):
        print("Render_Start")
        self.rend = True
        self.fig = plt.figure(num=self.drone_name,figsize=(12, 4))
        #fig, axes = plt.subplots(nrows=2, ncols=3, figsize=plt.figaspect(1/2))
        self.update_viwer()
        
    def update_viwer(self):
        plt.clf()
        colors = ['white', 'blue', 'red', 'yellow']
        cmap = matplotlib.colors.ListedColormap(colors)
        for i in range(1, 8):
            self.fig.add_subplot(2, 4, i)
            img = self.grid_map.global_map[:,:,(i-1)*2].copy()
            #self.fig.subplots(2, 4).flat[i-1].imshow(img,cmap)
            plt.imshow(img,cmap)
        self.fig.add_subplot(2, 4, 8)
        plt.imshow(self.grid_map.local_map[:,:,10],cmap)
        #self.fig.subplots(2, 4).flat[7].imshow(self.grid_map.local_map[:,:,10],cmap)
        plt.draw()
        plt.pause(0.000001)
        if self.episodic_length%500 == 0:
            img_dir = 'log/' + self.drone_name +str(self.episodic_length) +'.jpg'
            plt.savefig(img_dir)
        #print("update_viwer")

    def manual_control(self):
        #start joystick control thread
        pass

        