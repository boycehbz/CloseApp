'''
 @FileName    : Visualization3D.py
 @EditTime    : 2021-07-07 15:25:11
 @Author      : Buzhen Huang
 @Email       : hbz@seu.edu.cn
 @Description : 
'''

import open3d as o3d
import time

class Visualization(object):
    def __init__(self) -> None:
        # set viewer
        self.viewer = o3d.visualization.Visualizer() #O3DVisualizer Visualizer
        window_size = 600
        self.viewer.create_window(
            width=window_size + 1, height=window_size + 1,
            window_name='result'
        )

    def visualize_cameras(self, points, color):
        point_cloud1 = o3d.geometry.PointCloud()
        point_cloud2 = o3d.geometry.PointCloud()
        lineset = o3d.geometry.LineSet()
        for i in range(len(points)//2):
            point_cloud1.points = o3d.utility.Vector3dVector(points[i*2].reshape(-1,3))
            point_cloud2.points = o3d.utility.Vector3dVector(points[i*2+1].reshape(-1,3))

            lineset_one = lineset.create_from_point_cloud_correspondences(point_cloud1, point_cloud2, [(0, 0)])
            self.viewer.add_geometry(lineset_one.paint_uniform_color(color))
            self.viewer.poll_events()

    def visualize_points(self, points, color):
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        self.viewer.add_geometry(point_cloud)

        # for i in range(len(points)):

        point_cloud.points = o3d.utility.Vector3dVector(points)
        point_cloud.paint_uniform_color(color)
        # o3d.visualization.draw_geometries([point_cloud])
        self.viewer.update_geometry(point_cloud)
        # self.viewer.update_geometry(mesh_gt)
        # time.sleep(1)
        self.viewer.poll_events()

    def show(self):
        self.viewer.poll_events()