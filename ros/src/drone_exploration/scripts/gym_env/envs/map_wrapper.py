from cv2 import rotate
import numpy as np
import math
import time
import yaml
from transforms3d import quaternions

# STEP 1 坐标转换，用一个角点作为原点
# STEP 2 区域离散化，将空间变成矩阵代表的grid
# STEP 3 搞清楚插入的点云numpy矩阵是什么东西 -- 以传感器光心为原点，垂直传感器建立坐标系的h*w个三维点坐标，如果depth为nan，则坐标为nan
# STEP 3.5 建立临时的地图
# STEP 4 插入点云(增加对应区域为occupy的概率) -- 参考如何概率插入点云地图
# STEP 5 对应插入的occupy与传感器位置连线设为空，降低此处为occupy的概率
# STEP 6 合并临时地图和全局地图


#distortion
#-0.000591 0.000519 0.000001 -0.000030 0.000000

def pointcloud_from_depth(depth, fx, fy, cx, cy):
    """convert the depth image into pointcloud
       Notice: the unit of the coordinates is same as 1000 * z / 256 of depth image
    Args:
        depth (np.array): shape w*h
        fx (float): Camera internal parameter
        fy (float): Camera internal parameter
        cx (float): Camera internal parameter
        cy (float): Camera internal parameter
    Returns:
        np.array: shape w*h*3, Cartesian coordinates of the pointcloud
    """
    print("orgpointcloud_from_depth",time.time())
    assert depth.dtype.kind == "f", "depth must be float and have meter values"
    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
    valid = ~np.isnan(depth)
    z = np.where(valid, depth, np.nan)
    x = np.where(valid, z * (c - cx) / fx, 0)
    y = np.where(valid, z * (r - cy) / fy, 0)
    pc = np.dstack((x, y, z)).squeeze()
    pc = pc.reshape((pc.shape[0]*pc.shape[1],3))
    print("orgpointcloud_from_depth",time.time())
    return pc

def generate_pointcloud(depth,fx, fy, cx, cy):
    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
    valid = (depth > 0) & (depth < 255)
    z = 1000 * np.where(valid, depth / 256.0, np.nan)
    x = np.where(valid, z * (c - cx) / fx, 0)
    y = np.where(valid, z * (r - cy) / fy, 0)
    return np.dstack((x, y, z))


class GridMap():
    def __init__(self, drone_name, config_path: str = '../configs/env_config_simple.yaml') -> None:
        #读配置文件
        yaml_instream = open(config_path,'r',encoding='utf-8')
        yaml_config = yaml_instream.read()
        config = yaml.safe_load(yaml_config)
        self.camera_pram = config['drone']['camera_pram']
        self.depth_max = config['drone']['depth_max']
        self.global_range = np.array(config['map']['global_map_range'])
        self.world_orgin = np.array(config['map']['world_orgin'][drone_name])
        self.loacl_map_range =  np.array(config['map']['loacl_map_range'])
        self.resolution = config['map']['resolution']
        yaml_instream.close()

        #建立地图，全局地图，临时地图，局部地图，路径坐标
        self.global_map_shape = np.around(self.global_range/self.resolution).astype(np.int)
        self.global_map = np.ones(shape=(self.global_map_shape),dtype=np.uint8)
        self.tmp_map = np.ones(shape=(self.global_map_shape),dtype=np.uint8)
        self.loacl_map_shape = np.around(self.loacl_map_range/self.resolution).astype(np.int)
        self.local_map = np.ones(self.loacl_map_shape,dtype=np.uint8) 
        self.path_list = []

    def init_with_file(self, file_path, position):
        #????传感器的位姿是右手坐标系，而GroundTruth是左手坐标系
        position[1] = -position[1]
        position[2] = -position[2]
        position_round = np.around((position+self.world_orgin)/self.resolution).astype(np.int)
        print(file_path)
        self.path_list = []
        self.path_list.append(position_round)
        self.global_map = np.load(file_path)
        self._update_local_map(position_round)

    def _quaternion_to_euler(self,quaternionr):
        """将四元数转换为欧拉角
        Args:
            quaternionr (airsim.Quaternionr()): x/y/z/w_val 
        Returns:
            float: r,p,y degree
        """
        #虚幻引擎中的左手坐标系的旋转转换成右手坐标系，yz轴反向即可
        #传感器的位姿是右手坐标系，而GroundTruth是左手坐标系
        x = quaternionr.x_val
        y = -quaternionr.y_val
        z = -quaternionr.z_val
        # y = quaternionr.y_val
        # z = quaternionr.z_val
        w = quaternionr.w_val
        r = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        r = r / math.pi * 180
        p = math.asin(2 * (w * y - z * x))
        p = p / math.pi * 180
        y = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        y = y / math.pi * 180
        return r,p,y

    def _quaternion2RotationMatrix(self, quaternion):
        #虚幻引擎中的左手坐标系的旋转转换成右手坐标系，yz轴反向即可
        #??????传感器的位姿是右手坐标系，而GroundTruth是左手坐标系
        x, y, z, w = quaternion.x_val, -quaternion.y_val, -quaternion.z_val, quaternion.w_val
        #x, y, z, w = quaternion.x_val, quaternion.y_val, quaternion.z_val, quaternion.w_val
        rot_matrix00 = 1 - 2 * y * y - 2 * z * z
        rot_matrix01 = 2 * x * y - 2 * w * z
        rot_matrix02 = 2 * x * z + 2 * w * y
        rot_matrix10 = 2 * x * y + 2 * w * z
        rot_matrix11 = 1 - 2 * x * x - 2 * z * z
        rot_matrix12 = 2 * y * z - 2 * w * x
        rot_matrix20 = 2 * x * z - 2 * w * y
        rot_matrix21 = 2 * y * z + 2 * w * x
        rot_matrix22 = 1 - 2 * x * x - 2 * y * y
        return np.asarray([
            [rot_matrix00, rot_matrix01, rot_matrix02],
            [rot_matrix10, rot_matrix11, rot_matrix12],
            [rot_matrix20, rot_matrix21, rot_matrix22]
        ], dtype=np.float64)

    def _depth_dedistortion(self, pointDepth, f, depth_max):
        """深度图去畸变，仅采用内参中的f=fx=fy来进行去畸变
            限制深度范围，超范围的置0
        Args:
            pointDepth (W*H的np.array): 深度图
            f (float): 相机内参
            depth_max (float): 深度范围

        Returns:
            W*H的np.array: 去畸变之后的深度图
        """
        #print("depth_dedistortion",time.time())
        H = pointDepth.shape[0]
        W = pointDepth.shape[1]
        i_c = float(H) / 2 - 1
        j_c = float(W) / 2 - 1
        columns, rows = np.meshgrid(np.linspace(0, W-1, num=W), np.linspace(0, H-1, num=H))
        DistanceFromCenter = ((rows - i_c)**2 + (columns - j_c)**2)**(0.5)
        PlaneDepth = pointDepth / (1 + (DistanceFromCenter / f)**2)**(0.5)
        PlaneDepth[PlaneDepth > depth_max] = 0
        return PlaneDepth

    def _pointcloud_from_depth(self, depth, fx, fy, cx, cy):
        """从深度图转换为点云

        Args:
            depth (W*H的np.array): 去畸变之后的深度图
            fx (float): 相机内参fx
            fy (float): 相机内参fy
            cx (float): 相机内参cx
            cy (float): 相机内参cy

        Returns:
            np.array: 3*Len 的点云
        """
        #print("pointcloud_from_depth",time.time())
        rows, cols = depth.shape
        c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
        valid = ~np.isnan(depth)
        z = np.where(valid, depth, np.nan)
        x = (np.multiply(z,(cx-c))/fx).ravel()
        y = (np.multiply(z,(cy-r))/fy).ravel()
        z = z.ravel()
        pc = np.dstack((z, x, y))[0].T
        return pc

    def _update_grid(self, pointcloud, position, quaternionr):
        """利用点云数据，位置，姿态数据，来更新全局地图

        Args:
            pointcloud (np.array): 3*Len 的点云
            position (np.array,float): 原始的传感器相对于全局地图的位置
            quaternionr (airsim.Quaternionr()): 姿态四元数
        """
        #print("_update_grid",time.time())
        self.tmp_map = np.ones(shape=(self.global_map_shape),dtype=np.uint8)

        #STEP1 从 position，quaternionr，world_orgin生成从传感器坐标系到地图坐标系的转换
        rotate_matrix = self._quaternion2RotationMatrix(quaternionr)
        position_41 = np.expand_dims(np.hstack((position+self.world_orgin, np.array([1]))),axis=1)
        trans_matrix = np.hstack((np.vstack((rotate_matrix, np.array([0,0,0]))),position_41))

        #STEP2 将pointcloud中的坐标从传感器坐标系下转换为地图坐标系下
        pointcloud = np.matmul(trans_matrix, np.vstack((pointcloud, np.ones((1,pointcloud.shape[1])))))[:3,:]
        pointcloud = np.around(pointcloud/self.resolution).astype(np.int)
        
        #STEP3 取出点云中的无效点（超出地图范围的点）
        #TODO Need To Speedup
        mask0 = np.less(pointcloud[0,:], self.global_map_shape[0])
        mask1 = np.less(pointcloud[1,:], self.global_map_shape[1])
        mask2 = np.less(pointcloud[2,:], self.global_map_shape[2])
        mask = np.logical_and(np.logical_and(mask0,mask1),mask2)
        mask0 = np.less(0,pointcloud[0,:])
        mask1 = np.less(0,pointcloud[1,:])
        mask2 = np.less(0,pointcloud[2,:])
        mask = np.logical_and(np.logical_and(np.logical_and(mask0,mask1),mask2),mask)
        pointcloud = pointcloud[:,mask]
        #np.savetxt("/home/mrmmm/DRL_Exploration_With_Airsim/ros/src/drone_exploration/scripts/log/map/pointcloud—after_rotate_bisae.txt",pointcloud.T,fmt='%d')
        #print("pointcloud_after_rotate_biase_mask: ",pointcloud.shape)

        #STEP4 更新占用区域，并对于点云进行降采样和去重
        try:
            self.tmp_map[tuple(pointcloud)] = 3
            self.global_map[tuple(pointcloud)] = 3
        except(IndexError):
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!IndexError!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        pointcloud = np.dstack(np.nonzero(self.tmp_map==3))[0]


        #STEP5 更新空闲区域
        #TODO Need To Speedup
        position_round = np.around((position+self.world_orgin)/self.resolution).astype(np.int)
        for i in range(pointcloud.shape[0]):
            self._update_ray_discrete(position_round,pointcloud[i])
        free_uodate = np.dstack(np.nonzero(self.tmp_map==0))[0].T
        print("shape of free_uodate: ", free_uodate.shape)
        self.global_map[tuple(free_uodate)] = 0

    def _update_ray_discrete(self, begin, end):
        direction = end - begin
        x_0 = np.arange(begin[0],end[0]).astype(np.int) - begin[0]
        y_0 = np.arange(begin[1],end[1]).astype(np.int) - begin[1]
        z_0 = np.arange(begin[2],end[2]).astype(np.int) - begin[2]
        #print("_update_ray_discrete:" , x_0,y_0,z_0,begin,end)
        # use x linespace
        if direction[0]:
            x_x = x_0 + begin[0]
            y_x = np.around(x_0 * (direction[1]/direction[0]) + begin[1]).astype(np.int)
            z_x = np.around(x_0 * (direction[2]/direction[0]) + begin[2]).astype(np.int)
            x_free = np.dstack((x_x, y_x, z_x))[0]
            self.tmp_map[tuple(x_free.T)] = 0
        # use y linespace
        if direction[1]:
            y_y = y_0 + begin[1]
            x_y = np.around(y_0 * (direction[0]/direction[1]) + begin[0]).astype(np.int)
            z_y = np.around(y_0 * (direction[2]/direction[1]) + begin[2]).astype(np.int)
            y_free = np.dstack((x_y, y_y, z_y))[0]
            self.tmp_map[tuple(y_free.T)] = 0
        # use z linespace
        if direction[2]:
            z_z = z_0 + begin[2]
            x_z = np.around(z_0 * (direction[0]/direction[2]) + begin[0]).astype(np.int)
            y_z = np.around(z_0 * (direction[1]/direction[2]) + begin[1]).astype(np.int)
            z_free = np.dstack((x_z, y_z, z_0))[0]
            self.tmp_map[tuple(z_free.T)]

    def _update_local_map(self,position_round):
        # 限制local范围
        self.local_map = np.ones(self.loacl_map_shape,dtype=np.uint8) * 3
        region_start = np.around(position_round - self.loacl_map_shape/2).astype(np.int)
        region_end = region_start + self.loacl_map_shape
        global_region_start = np.where(region_start>0,region_start,0)
        global_region_end = np.where(region_end<self.global_map_shape,region_end,self.global_map_shape)
        local_region_start = np.where(region_start>0,0,-region_start)
        local_region_end = np.where(region_end<self.global_map_shape,self.loacl_map_shape,self.loacl_map_shape - (region_end-self.global_map_shape))
        # print("position_round: ",position_round)
        # print("region_end: ", region_end)
        # print("global_region_start,global_region_end: ",global_region_start,global_region_end)
        # print("local_region_start,local_region_end: ",local_region_start,local_region_end)
        try:
            self.local_map[local_region_start[0]:local_region_end[0],local_region_start[1]:local_region_end[1],local_region_start[2]:local_region_end[2]] = self.global_map[global_region_start[0]:global_region_end[0],global_region_start[1]:global_region_end[1],global_region_start[2]:global_region_end[2]]
        except:
            print("ERROR: _update_local_map")
            print("position_round: ",position_round)
            print("region_end: ", region_end)
            print("global_region_start,global_region_end: ",global_region_start,global_region_end)
            print("local_region_start,local_region_end: ",local_region_start,local_region_end)

    def update_map_depth(self,depth, position, quaternionr):
        #STEP1 位置坐标系转换，从虚幻引擎的左手坐标系转换为右手坐标系；?????传感器的位姿是右手坐标系，而GroundTruth是左手坐标系
        position[1] = -position[1]
        position[2] = -position[2]

        #STEP2 采用深度图和位姿更新占用区域和空闲区域
        depth = self._depth_dedistortion(depth, self.camera_pram['fx'], self.depth_max)
        pointcloud = self._pointcloud_from_depth(depth, self.camera_pram['fx'], self.camera_pram['fy'], self.camera_pram['cx'], self.camera_pram['cy'])
        self._update_grid(pointcloud, position, quaternionr)

        #STEP3 采用历史路径和当前位置更新地图中的路径
        position_round = np.around((position+self.world_orgin)/self.resolution).astype(np.int)
        if len(self.path_list)==0 or (not (np.any(np.all(position_round == np.array(self.path_list), axis=1)))):
            self.path_list.append(position_round)
        self.global_map[tuple(np.array(self.path_list).T)] = 2

        #STEP4更新局部地图
        self._update_local_map(position_round)
        print("update_map done",time.time())

    def update_map_pcl(self, pointcloud, position, quaternionr):
        #STEP1 位置坐标系转换，从虚幻引擎的左手坐标系转换为右手坐标系；??????传感器的位姿是右手坐标系，而GroundTruth是左手坐标系
        position[1] = -position[1]
        position[2] = -position[2]

        #STEP2 采用点云和位姿更新占用区域和空闲区域
        pointcloud = pointcloud.T
        self._update_grid(pointcloud, position, quaternionr)

        #STEP3 采用历史路径和当前位置更新地图中的路径
        position_round = np.around((position+self.world_orgin)/self.resolution).astype(np.int)
        if len(self.path_list)==0 or (not (np.any(np.all(position_round == np.array(self.path_list), axis=1)))):
            self.path_list.append(position_round)
        self.global_map[tuple(np.array(self.path_list).T)] = 2

        #STEP4更新局部地图
        self._update_local_map(position_round)
        print("update_map done",time.time())

def main():
    pass


if __name__ == "__main__":
    main()