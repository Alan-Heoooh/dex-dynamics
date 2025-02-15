import copy
import os
import sys

import rclpy
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster
from transforms3d.quaternions import *
import yaml
import readchar


fixed_frame = 'marker'
# num_cams = 2
tune_idx = [1]  # set the index of the camera to tune

data_dir = '/home/coolbot/Documents/git/dex-dynamics/calibration'
with open(os.path.join(data_dir, '3cameras_link_1.yml'), 'r') as f:
    camera_pose_dict = yaml.load(f, Loader=yaml.FullLoader)
path_to_save = os.path.join(data_dir, '3cameras_link_1.yml')

# step side for adjusting
pos_stride = 0.02 / 40
rot_stride = 0.02 / 10


def pos_quat_to_matrix(pos, quat):
    assert len(pos) == 3, 'position should be xyz'
    rot = quat2mat(quat)
    pos = np.expand_dims(pos, 1)
    matrix = np.concatenate((np.concatenate((rot, pos), axis=1), [[0, 0, 0, 1]]))
    return matrix


def main(args=None):
    rclpy.init(args=args)

    node = rclpy.create_node('cam_pose_tuner')

    static_br = StaticTransformBroadcaster(node)
    static_ts_list = []
    for i in ['0', '1',  '3']:
        static_ts = TransformStamped()
        static_ts.header.frame_id = fixed_frame
        static_ts.child_frame_id = f"cam_{i}_link"

        static_ts.transform.translation.x = camera_pose_dict[f"cam_{i}"]["position"][0]
        static_ts.transform.translation.y = camera_pose_dict[f"cam_{i}"]["position"][1]
        static_ts.transform.translation.z = camera_pose_dict[f"cam_{i}"]["position"][2]

        static_ts.transform.rotation.x = camera_pose_dict[f"cam_{i}"]["orientation"][1]
        static_ts.transform.rotation.y = camera_pose_dict[f"cam_{i}"]["orientation"][2]
        static_ts.transform.rotation.z = camera_pose_dict[f"cam_{i}"]["orientation"][3]
        static_ts.transform.rotation.w = camera_pose_dict[f"cam_{i}"]["orientation"][0]

        static_ts_list.append(static_ts)

    static_br.sendTransform(static_ts_list)

    pcd_trans_vec = [0.0, 0.0, 0.0]
    pcd_rot_vec = [0.0, 0.0, 0.0]

    camera_pose_dict_new = copy.deepcopy(camera_pose_dict)

    br = TransformBroadcaster(node)
    # rate = node.create_rate(30)

    # record initial relative transformation
    ref_cam_mat = None
    cam2cam_relative_mats = {}
    for cam_idx in tune_idx:
        ref_cam_init_mat = pos_quat_to_matrix(camera_pose_dict[f"cam_{tune_idx[0]}"]["position"],
                                              camera_pose_dict[f"cam_{tune_idx[0]}"]["orientation"])
        cam_init_mat = pos_quat_to_matrix(camera_pose_dict[f"cam_{cam_idx}"]["position"],
                                          camera_pose_dict[f"cam_{cam_idx}"]["orientation"])
        cam_to_ref = np.linalg.inv(ref_cam_init_mat) @ cam_init_mat
        cam2cam_relative_mats[cam_idx] = cam_to_ref
        # import ipdb; ipdb.set_trace()
        # print(ref_cam_init_mat @ cam_to_ref, cam_init_mat)
        #
        # ref_cam_mat = ref_cam_init_mat
        # cam_mat = ref_cam_mat @ cam2cam_relative_mats[cam_idx]
        # cam_ori_cur = mat2quat(cam_mat[:3, :3])a
        # cam_pos_cur = cam_mat[:3, -1].tolist()
        #
        # print(cam_ori_cur, camera_pose_dict[f"cam_{tune_idx[cam_idx]}"]["position"])
        # print(cam_ori_cur, camera_pose_dict[f"cam_{tune_idx[cam_idx]}"]["orientation"])
        # print(np.linalg.inv(cam_init_mat) @ ref_cam_init_mat @ cam_to_ref == cam_init_mat)

    save = False
    while rclpy.ok():
        key = readchar.readkey()         # this actually blocks the thread
        print(key)

        if key == 'w':
            pcd_trans_vec[0] += pos_stride
        elif key == 'x':
            pcd_trans_vec[0] -= pos_stride
        elif key == 'a':  # y axis translation
            pcd_trans_vec[1] += pos_stride
        elif key == 'd':
            pcd_trans_vec[1] -= pos_stride
        elif key == 'q':
            pcd_trans_vec[2] += pos_stride
        elif key == 'z':
            pcd_trans_vec[2] -= pos_stride
        elif key == '1':
            pcd_rot_vec[0] += rot_stride
        elif key == '2':
            pcd_rot_vec[0] -= rot_stride
        elif key == '3':
            pcd_rot_vec[1] += rot_stride
        elif key == '4':
            pcd_rot_vec[1] -= rot_stride
        elif key == '5':        # z axis rotation
            pcd_rot_vec[2] += rot_stride
        elif key == '6':
            pcd_rot_vec[2] -= rot_stride
        elif key == 'm':
            with open(path_to_save, 'w') as f:
                yaml.dump(camera_pose_dict_new, f)
            print(f'saved to path: {path_to_save}')
        elif key == 'b':
            break

        pcd_ori_world = qmult(qmult(qmult(axangle2quat([1, 0, 0], pcd_rot_vec[0]),
                                          axangle2quat([0, 1, 0], pcd_rot_vec[1])),
                                    axangle2quat([0, 0, 1], pcd_rot_vec[2])),
                              [1.0, 0.0, 0.0, 0.0])

        # for each camera, compute the updated pose
        for i, cam_idx in enumerate(tune_idx):
            if i == 0:
                # this is the orientation of reference camera
                cam_pos_init = camera_pose_dict[f"cam_{cam_idx}"]["position"]
                cam_ori_init = camera_pose_dict[f"cam_{cam_idx}"]["orientation"]

                cam_pos_cur = np.array(cam_pos_init) + np.array(pcd_trans_vec)
                cam_pos_cur = [float(x) for x in cam_pos_cur]

                cam_ori_cur = qmult(pcd_ori_world, cam_ori_init)
                cam_ori_cur = [float(x) for x in cam_ori_cur]
                print(f"{cam_idx}: Pos: {cam_pos_cur}\nOri: {cam_ori_cur}")
                ref_cam_mat = pos_quat_to_matrix(cam_pos_cur, cam_ori_cur)
            # else:
            #     # other cameras follow
            #     cam_mat = ref_cam_mat @ cam2cam_relative_mats[cam_idx]
            #     cam_ori_cur = mat2quat(cam_mat[:3, :3])
            #     cam_pos_cur = cam_mat[:3, -1].tolist()

            #     cam_ori_cur = [float(x) for x in cam_ori_cur]
            #     cam_pos_cur = [float(x) for x in cam_pos_cur]
            #     print(f"{cam_idx} follows {tune_idx[0]}: Pos: {cam_pos_cur}\nOri: {cam_ori_cur}")

            # broadcast transformations
            transform_stamped = TransformStamped()
            transform_stamped.header.stamp = node.get_clock().now().to_msg()
            transform_stamped.header.frame_id = fixed_frame
            transform_stamped.child_frame_id = f"cam_{cam_idx}_link"

            transform_stamped.transform.translation.x = cam_pos_cur[0]
            transform_stamped.transform.translation.y = cam_pos_cur[1]
            transform_stamped.transform.translation.z = cam_pos_cur[2]

            transform_stamped.transform.rotation.x = cam_ori_cur[1]
            transform_stamped.transform.rotation.y = cam_ori_cur[2]
            transform_stamped.transform.rotation.z = cam_ori_cur[3]
            transform_stamped.transform.rotation.w = cam_ori_cur[0]

            br.sendTransform(transform_stamped)

            camera_pose_dict_new[f"cam_{cam_idx}"]["position"] = cam_pos_cur
            camera_pose_dict_new[f"cam_{cam_idx}"]["orientation"] = cam_ori_cur
            camera_pose_dict_new[f"cam_{cam_idx}"]['transformation'] = \
                pos_quat_to_matrix(cam_pos_cur, cam_ori_cur).tolist()

        # rate.sleep()


if __name__ == '__main__':
    main()
