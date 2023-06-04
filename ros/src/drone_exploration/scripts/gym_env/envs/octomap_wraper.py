#!/usr/bin/env python

import glooey
import imgviz
import numpy as np
import pyglet
import trimesh
import trimesh.transformations as tf
import trimesh.viewer
import airsim
import time

import octomap


def pointcloud_from_depth(depth, fx, fy, cx, cy):
    assert depth.dtype.kind == "f", "depth must be float and have meter values"

    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
    valid = ~np.isnan(depth)
    z = np.where(valid, depth, np.nan)
    x = np.where(valid, z * (c - cx) / fx, np.nan)
    y = np.where(valid, z * (r - cy) / fy, np.nan)
    pc = np.dstack((x, y, z))

    return pc


def labeled_scene_widget(scene, label):
    vbox = glooey.VBox()
    vbox.add(glooey.Label(text=label, color=(255, 255, 255)), size=0)
    vbox.add(trimesh.viewer.SceneWidget(scene))
    return vbox


def visualize(
    occupied, empty, K, width, height, rgb, pcd, mask, resolution, aabb
):
    window = pyglet.window.Window(
        width=int(640 * 0.9 * 3), height=int(480 * 0.9)
    )

    @window.event
    def on_key_press(symbol, modifiers):
        if modifiers == 0:
            if symbol == pyglet.window.key.Q:
                window.on_close()

    gui = glooey.Gui(window)
    hbox = glooey.HBox()
    hbox.set_padding(5)

    camera = trimesh.scene.Camera(
        resolution=(width, height), focal=(K[0, 0], K[1, 1])
    )
    camera_marker = trimesh.creation.camera_marker(camera, marker_height=0.1)

    # initial camera pose
    camera_transform = np.array(
        [
            [0.73256052, -0.28776419, 0.6168848, 0.66972396],
            [-0.26470017, -0.95534823, -0.13131483, -0.12390466],
            [0.62712751, -0.06709345, -0.77602162, -0.28781298],
            [0.0, 0.0, 0.0, 1.0],
        ],
    )

    aabb_min, aabb_max = aabb
    bbox = trimesh.path.creation.box_outline(
        aabb_max - aabb_min,
        tf.translation_matrix((aabb_min + aabb_max) / 2),
    )

    # geom = trimesh.PointCloud(vertices=pcd[mask], colors=rgb[mask])
    # scene = trimesh.Scene(camera=camera, geometry=[bbox, geom, camera_marker])
    # scene.camera_transform = camera_transform
    # hbox.add(labeled_scene_widget(scene, label="pointcloud"))

    print("occupied")
    geom = trimesh.voxel.ops.multibox(
        occupied, pitch=resolution, colors=[1.0, 0, 0, 0.5]
    )
    scene = trimesh.Scene(camera=camera, geometry=[bbox, geom, camera_marker])
    scene.camera_transform = camera_transform
    hbox.add(labeled_scene_widget(scene, label="occupied"))

    print("empty")
    geom = trimesh.voxel.ops.multibox(
        empty, pitch=resolution, colors=[0.5, 0.5, 0.5, 0.5]
    )
    scene = trimesh.Scene(camera=camera, geometry=[bbox, geom, camera_marker])
    scene.camera_transform = camera_transform
    hbox.add(labeled_scene_widget(scene, label="empty"))

    gui.add(hbox)
    pyglet.app.run()


def main():
    # data = imgviz.data.arc2017()
    # camera_info = data["camera_info"]
    # K = np.array(camera_info["K"]).reshape(3, 3)
    # rgb = data["rgb"]
    # pcd = pointcloud_from_depth(
    #     data["depth"], fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2]
    # )
    K = np.array([[360.0,0,360.0],[0,360.0,240.0],[0,0,0]])
    client = airsim.MultirotorClient()
    resolution = 0.1
    octree = octomap.OcTree(resolution)
    start = time.time()
    for i in range(5000):
        # img_responses = client.simGetImages([airsim.ImageRequest("front_center_custom", airsim.ImageType.DepthPerspective,True, False),
        #                                     airsim.ImageRequest("front_center_custom", airsim.ImageType.Scene)],vehicle_name = "Drone_1")
        # img_response = img_responses[0]
        # img_responses1 = img_responses[1]
        # img1d = np.array(img_response.image_data_float, dtype=np.float)
        #     #在后面判断nan，就不归一化了
        #     #img1d[img1d > 255] = 255
        # img2d = np.reshape(img1d, (img_response.height, img_response.width))
        # pcd = pointcloud_from_depth(
        #     img2d, fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2]
        # )
        
        lidar_date = client.getLidarData(lidar_name='lidar_1',vehicle_name="Drone_1")
        points = np.array(lidar_date.point_cloud, dtype=np.dtype('f4'))
        pcd = np.reshape(points, (int(points.shape[0]/3), 3))

        #nonnan = ~np.isnan(pcd).any(axis=2)
        #mask = np.less(pcd[:, :, 2], 2)
        nonnan = ~np.isnan(pcd).any(axis=1)
        pcd = pcd[nonnan]
        mask0 = np.less(pcd[:,0], 0.2)
        mask1 = np.less(pcd[:,1], 0.2)
        mask2 = np.less(pcd[:,2], 0.2)
        mask = np.logical_and(np.logical_and(mask0,mask1),mask2)
        pcd = pcd[mask,:]
        print("pcd",pcd.shape)
        print("mask0",mask0.shape,mask0.sum())
        print("mask1",mask1.shape,mask1.sum())
        print("mask2",mask2.shape,mask1.sum())
        print("mask",mask.shape,mask.sum())
        print("pcd.shape:", type(pcd), pcd.shape, pcd.dtype)
        octree.insertPointCloud(
            pointcloud=pcd.astype(np.float64),
            origin=np.array([lidar_date.pose.position.x_val, lidar_date.pose.position.y_val, lidar_date.pose.position.z_val], dtype=np.double),
            #origin=np.array(lidar_date.pose.position.to_numpy_array(), dtype=float),
            maxrange=2,
        )
        print(time.time())
    print("Instert time: ", (time.time() - start)/1000)
    octree.updateInnerOccupancy()
    occupied, empty = octree.extractPointCloud()
    print("occupied:",type(occupied),occupied.shape,occupied)
    print("empty:",type(empty),empty.shape, empty)

    aabb_min = octree.getMetricMin()
    aabb_max = octree.getMetricMax()

    visualize(
        occupied=occupied,
        empty=empty,
        K=K,
        #width=img_response.width,
        #height=img_response.height,
        #rgb=img_responses1.image_data_uint8,
        width=720,
        height=480,
        rgb=None,
        pcd=pcd,
        mask=mask,
        resolution=resolution,
        aabb=(aabb_min, aabb_max),
    )


if __name__ == "__main__":
    main()