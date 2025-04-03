from wis3d import Wis3D
import trimesh
import sapien
import numpy as np
import warnings
import transforms3d


class SAPIENKinematicsModelStandalone:
    def __init__(self, urdf_path, srdf_path=None):
        self.engine = sapien.Engine()

        self.scene = self.engine.create_scene()

        loader = self.scene.create_urdf_loader()
        # loader.scale = 10
        builder = loader.load_file_as_articulation_builder(urdf_path, srdf_path)
        # builder = loader.load(urdf_path)

        self.robot: sapien.Articulation = builder.build(fix_root_link=True)

        self.robot.set_pose(sapien.Pose())

        self.robot.set_qpos(np.zeros(self.robot.dof))

        self.scene.step()

        self.model: sapien.PinocchioModel = self.robot.create_pinocchio_model()

    def compute_forward_kinematics(self, qpos, link_index) -> sapien.Pose:
        # print('left 6!!!!')
        if len(qpos) != self.robot.dof:
            warnings.warn("qpos length not match")
        qpos = np.array(qpos).tolist() + [0] * (self.robot.dof - len(qpos))
        # print('in fk', qpos, link_index)
        # self.robot.set_qpos(qpos)
        self.model.compute_forward_kinematics(np.array(qpos))

        return self.model.get_link_pose(link_index)
        # for i in self.robot.links:
        #     if i.name == 'Link_left6':
        #         return i.pose

    #
    def compute_inverse_kinematics(
        self, link_index, pose, initial_qpos, *args, **kwargs
    ):
        if len(initial_qpos) != self.robot.dof:
            warnings.warn("initial_qpos length not match")
        initial_qpos = np.array(initial_qpos).tolist() + [0] * (
            self.robot.dof - len(initial_qpos)
        )

        return self.model.compute_inverse_kinematics(
            link_index, pose, initial_qpos=initial_qpos, *args, **kwargs
        )

    def release(self):
        self.scene = None

        self.engine = None


class Wis3D(Wis3D):
    urdf_caches = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def add_robot(
        self,
        urdf_path,
        qpos=None,
        Tw_w2B=None,
        add_local_coord=False,
        name="",
        return_joint_mesh=False,
    ):
        """
        add robot.
        Parameters
        ----------
        urdf_path: str. path to urdf file
        qpos: len(active_joints)-dim ndarray. radian. default to be 0. If len(qpos) < len(active_joints), the rest will be filled with 0.
        Tw_w2B: world to base transformation.
        add_local_coord: whether to visualize per-link local coordinate system.
        name: keep this empty unless you are visualizing multiple robots.

        Returns
        -------

        """
        from ..utils import utils_3d

        if not self.enable:
            return
        try:
            import sapien
        except ImportError:
            raise ImportError("Please install sapien first.")
        if urdf_path not in Wis3D.urdf_caches:
            Wis3D.urdf_caches[urdf_path] = SAPIENKinematicsModelStandalone(urdf_path)
        if Tw_w2B is None:
            Tw_w2B = np.eye(4)

        sk: SAPIENKinematicsModelStandalone = Wis3D.urdf_caches[urdf_path]
        links = sk.robot.get_links()
        if qpos is None:
            warnings.warn("qpos is not provided, using default qpos.")
            qpos = np.zeros(len(sk.robot.get_active_joints()))
        if len(qpos) < len(sk.robot.get_active_joints()):
            warnings.warn(
                f"qpos is not complete {len(qpos)} < {len(sk.robot.get_active_joints())}, filling the rest with 0."
            )
            qpos = np.asarray(qpos).tolist() + [0] * (
                len(sk.robot.get_active_joints()) - len(qpos)
            )
        local_pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]) * 0.05
        joint_mesh = None
        for link_index, link in enumerate(links):
            link_name = link.name
            pq = sk.compute_forward_kinematics(qpos, link_index)

            pose = utils_3d.Rt_to_pose(transforms3d.quaternions.quat2mat(pq.q), pq.p)
            components = link.get_entity().get_components()
            for component in components:
                if isinstance(component, sapien.pysapien.render.RenderBodyComponent):
                    local_pose = component.render_shapes[
                        0
                    ].local_pose.to_transformation_matrix()
                    pose = pose @ local_pose
                    if hasattr(component.render_shapes[0], "filename"):
                        mesh_path = component.render_shapes[0].filename
                        mesh = trimesh.load(mesh_path, force="mesh")
                        add_name = link_name if name == "" else name + "_" + link_name
                        m = trimesh.Trimesh(
                            utils_3d.transform_points(mesh.vertices, Tw_w2B @ pose),
                            mesh.faces,
                        )
                        if return_joint_mesh:
                            if joint_mesh is None:
                                joint_mesh = m
                            else:
                                joint_mesh = trimesh.util.concatenate(joint_mesh, m)
                        self.add_mesh(m, name=add_name)
                        axis_in_base = utils_3d.transform_points(
                            local_pts, Tw_w2B @ pose
                        )
                        if add_local_coord:
                            self.add_coordinate_transformation(
                                Tw_w2B @ pose, name=f"{link_name}_coord"
                            )
                            # self.add_lines(axis_in_base[0], axis_in_base[1], name=f'{link_name}_x')
                            # self.add_lines(axis_in_base[0], axis_in_base[2], name=f'{link_name}_y')
                            # self.add_lines(axis_in_base[0], axis_in_base[3], name=f'{link_name}_z')
        if return_joint_mesh:
            return joint_mesh