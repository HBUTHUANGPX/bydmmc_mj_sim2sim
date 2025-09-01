import time
import mujoco.viewer
import mujoco
import numpy as np
import torch
import yaml

from typing import Union, List, Dict, Tuple
from scipy.spatial.transform import Rotation as R

import glfw
import os
import onnxruntime as ort
import copy
import math
import matplotlib.pyplot as plt
import queue
import threading
import pickle

from math_func import *
from motion_loader import MotionLoader
from video_recorder import VideoRecorder

np.set_printoptions(precision=16, linewidth=100, threshold=np.inf, suppress=True)

current_path = os.getcwd()


class cfg:
    simulator_dt = 0.002
    policy_dt = 0.02

    policy_type = "onnx"  # torch or onnx
    policy_path = (
        current_path
        + "/deploy_mujoco/deploy_policy/2025-08-29_14-15-44_h1_2_LF2_w1s1_v0_30000step.onnx"
    )
    mjcf_path = current_path + "/deploy_mujoco/assets/unitree_h1_2/h1_2.xml"
    motion_file = (
        current_path + "/deploy_mujoco/deploy_policy/artifacts/w1s1_h1_2:v0/motion.npz"
    )
    sim_data_filename = current_path + "/deploy_mujoco/deploy_policy/data.pkl"
    only_leg_flag = False  # True, False
    with_wrist_flag = True  # True, False

    ###############################
    # stiffness and damping param #
    ###############################
    leg_P_gains = [60, 220, 220, 320, 40, 40] * 2
    leg_D_gains = [1.5, 4, 4, 4, 2.0, 2.0] * 2

    torso_P_gains = [300.0]
    torso_D_gains = [3.5]

    arm_P_gains = [240.0, 240.0, 240.0, 160.0, 160.0, 160.0, 160.0] * (2)
    arm_D_gains = [4.0, 4.0, 4.0, 3.0, 3.0, 3.0, 3.0] * (2)

    #####################
    # joint default pos #
    #####################
    leg_default_pos = [0.0, -0.15, 0.0, 0.3, -0.15, 0.0] * (2)
    torso_default_pos = [0.0]
    arm_default_pos = [0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0] + [
        0.2,
        -0.2,
        0.0,
        0.6,
        0.0,
        0.0,
        0.0,
    ]

    ########################
    # joint maximum torque #
    ########################
    leg_tq_max = [200.0, 200.0, 200.0, 300.0, 50.0, 50.0] * (2)
    torso_tq_max = [200.0]
    arm_tq_max = [40, 40.0, 18.0, 18.0, 19.0, 19.0, 19.0] * (2)

    ################
    # action param #
    ################
    action_clip = 10.0
    action_scale = 0.25
    action_num = 0

    action_num = 12
    if not only_leg_flag:
        if not with_wrist_flag:
            action_num += 8 + 1
        else:
            action_num += 14 + 1
    print("action_num: ", action_num)
    #############
    # obs param #
    #############
    frame_stack = 1
    num_single_obs = 150

    ####################
    # motion play mode #
    ####################
    """
     if motion_play is true, robots in mujoco will set 
     qpos and qvel through the retargeting dataset 
    """
    motion_play = False  # False, True
    """
    if motion_play is true and sim_motion_play is true,
    robots in mujoco will set qpos and qvel through the 
    dataset recorded in isaac sim
    """
    sim_motion_play = False  # False, True,

    ###########################################
    # Data conversion of isaac sim and mujoco #
    ###########################################
    isaac_sim_joint_name = [
        "left_hip_yaw_joint",
        "right_hip_yaw_joint",
        "torso_joint",
        "left_hip_pitch_joint",
        "right_hip_pitch_joint",
        "left_shoulder_pitch_joint",
        "right_shoulder_pitch_joint",
        "left_hip_roll_joint",
        "right_hip_roll_joint",
        "left_shoulder_roll_joint",
        "right_shoulder_roll_joint",
        "left_knee_joint",
        "right_knee_joint",
        "left_shoulder_yaw_joint",
        "right_shoulder_yaw_joint",
        "left_ankle_pitch_joint",
        "right_ankle_pitch_joint",
        "left_elbow_joint",
        "right_elbow_joint",
        "left_ankle_roll_joint",
        "right_ankle_roll_joint",
        "left_wrist_roll_joint",
        "right_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "right_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_wrist_yaw_joint",
    ]

    mujoco_joint_name = [
        "left_hip_yaw_joint",
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_yaw_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "torso_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ]  # [joint.name for joint in self.spec.joints][1:]

    isaac_sim_link_name = [
        "pelvis",
        "left_hip_yaw_link",
        "right_hip_yaw_link",
        "torso_link",
        "left_hip_pitch_link",
        "right_hip_pitch_link",
        "left_shoulder_pitch_link",
        "right_shoulder_pitch_link",
        "left_hip_roll_link",
        "right_hip_roll_link",
        "left_shoulder_roll_link",
        "right_shoulder_roll_link",
        "left_knee_link",
        "right_knee_link",
        "left_shoulder_yaw_link",
        "right_shoulder_yaw_link",
        "left_ankle_pitch_link",
        "right_ankle_pitch_link",
        "left_elbow_link",
        "right_elbow_link",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_wrist_roll_link",
        "right_wrist_roll_link",
        "left_wrist_pitch_link",
        "right_wrist_pitch_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
    ]


class simulator:
    policy: ort.InferenceSession

    def __init__(self):
        # Load robot model
        self.spec = mujoco.MjSpec.from_file(cfg.mjcf_path)
        self._rehandle_xml()
        self.m = self.spec.compile()
        # self.m = mujoco.MjModel.from_xml_path(cfg.mjcf_path)
        self.d = mujoco.MjData(self.m)
        self._scene = mujoco.MjvScene(self.m, 100000)
        print(f"Number of actuators: {self.m.nu}")

        self.m.opt.timestep = cfg.simulator_dt
        self.paused = False
        self._init_policy_conf()
        self._init_robot_conf()
        with open(cfg.sim_data_filename, "rb") as f:
            self.data_collection = pickle.load(f)
        self.data_queue = queue.Queue()
        self.change_id = 0
        self.video_recorder = VideoRecorder(
            path=current_path + "/deploy_mujoco/recordings",
            tag=None,
            video_name="video_0",
            fps=int(1 / cfg.policy_dt),
            compress=False,
        )

    def updata_isaac_sim_robot_state(self):
        self.data = self.data_collection[
            int(self.time_step[0] % len(self.data_collection))
        ]
        robot_ref_body_index = 3
        motion_ref_body_index = 7
        self.d.qpos[0:3] = self.data["robot.data.body_pos_w"].detach().cpu().numpy()[0]
        self.d.qpos[0] -= 1.25
        self.d.qpos[3:7] = self.data["robot.data.body_quat_w"].detach().cpu().numpy()[0]
        self.d.qpos[7:] = (
            self.data["robot.data.joint_pos"][self.isaac_sim2mujoco_index]
            .detach()
            .cpu()
            .numpy()
        )
        self.d.qvel[0:3] = (
            self.data["robot.data.body_lin_vel_w"].detach().cpu().numpy()[0]
        )
        self.d.qvel[3:6] = (
            self.data["robot.data.body_ang_vel_w"].detach().cpu().numpy()[0]
        )
        self.d.qvel[6:] = (
            self.data["robot.data.joint_vel"][self.isaac_sim2mujoco_index]
            .detach()
            .cpu()
            .numpy()
        )

        mujoco.mj_forward(self.m, self.d)
        ts = self.data["time_steps"].unsqueeze(0).detach().cpu().numpy()
        self.update_obs(ts)
        self._policy_reasoning()
        mj_obs = self.obs[0]
        critic_obs = self.data["critic_obs"]
        critic_obs = torch.cat([critic_obs[:63], critic_obs[189:]])
        is_obs = critic_obs.unsqueeze(0).detach().cpu().numpy()
        obs_err = mj_obs - is_obs[0]
        """
        print("=" * 80)
        print("motion joint pos command: ",obs_err[:27])
        print("motion joint vel command: ",obs_err[27:54])
        print("motion_ref_pos_b: ", obs_err[54:57])
        print("motion_ref_ori_b: ", obs_err[57:63])
        print("base_lin_vel: ", obs_err[63:66])
        print("base_ang_vel: ", obs_err[66:69])
        print("joint_pos: ", obs_err[69:96])
        print("joint_vel: ", obs_err[96:123])
        print("actions: ", obs_err[123:150])
        """
        return ts

    def motion_play(self):
        self.d.qpos[0:3] = (
            self.motion.body_pos_w[self.time_step, 0, :].detach().cpu().numpy()
        )
        q = self.motion.body_quat_w[self.time_step, 0, :].detach().cpu().numpy()[0, :]
        self.d.qpos[3:7] = q
        self.d.qpos[7 : 7 + len(self.default_pos)] = (
            self.motion.joint_pos[self.time_step].detach().cpu().numpy()
        )[:, self.isaac_sim2mujoco_index]
        self.d.qvel[0:3] = (
            self.motion.body_lin_vel_w[self.time_step, 0, :].detach().cpu().numpy()
        )
        self.d.qvel[3:6] = (
            self.motion.body_ang_vel_w[self.time_step, 0, :].detach().cpu().numpy()
        )
        self.d.qvel[6 : 6 + len(self.default_pos)] = (
            self.motion.joint_vel[self.time_step]
            .detach()
            .cpu()
            .numpy()[:, self.isaac_sim2mujoco_index]
        )
        mujoco.mj_forward(self.m, self.d)

    def run(self):
        save_data_flag = 1
        self.counter = 0
        self.d.qpos[7 : 7 + len(self.default_pos)] = self.default_pos
        self.d.qpos[2] = 1.03
        mujoco.mj_forward(self.m, self.d)
        self.target_dof_pos = self.default_pos.copy()[: self.action_num]
        self.phase = 0
        # self.viewer = mujoco_viewer.MujocoViewer(self.m, self.d)
        if save_data_flag:
            i = 0
            if os.path.exists("data.csv"):
                os.remove("data.csv")
        self.viewer = mujoco.viewer.launch_passive(
            self.m, self.d, key_callback=self.key_callback
        )
        self.renderer = mujoco.renderer.Renderer(self.m, height=480, width=640)
        self.init_vel_geom(
            "Goal Vel: x: {:.2f}, y: {:.2f}, yaw: {:.2f},force_z:{:.2f}".format(
                self.cmd[0], self.cmd[1], self.cmd[2], 0.0
            )
        )
        self.prev_qpos = self.d.qpos
        # plot_thread = threading.Thread(target=self.plot_data, args=(self.data_queue,))
        # plot_thread.daemon = True
        # plot_thread.start()

        first_flag = False
        while self.viewer.is_running():
            if not first_flag:
                first_flag = True
                # self.time_step[:] = self.updata_isaac_sim_robot_state()*1.0-1
                if cfg.motion_play and cfg.sim_motion_play:
                    self.time_step[:] = self.updata_isaac_sim_robot_state() * 1.0
                    self.motion_play()
                    self.time_step *= 0
                else:
                    self.motion_play()
                mujoco.mj_step(self.m, self.d)
                self.viewer.sync()
            self.policy_loop()
        print("stop")
        self.video_recorder.stop()

    def policy_loop(self):
        # print("="*(20))
        self.counter += 1
        # print(self.d.qvel[0])
        quat = self.d.qpos[3:7]
        omega = self.d.qvel[3:6]
        self.qpos = self.d.qpos[7 : 7 + self.action_num]
        self.P_n = self.qpos - self.default_pos[: self.action_num]
        self.V_n = self.d.qvel[6 : 6 + self.action_num]

        if self.time_step >= self.motion.time_step_total:
            self.time_step *= 0

        if cfg.motion_play:
            if cfg.sim_motion_play:
                self.updata_isaac_sim_robot_state()
            else:
                self.motion_play()
        else:
            self.update_obs(self.time_step)
            self.h2_action = self.h_action.copy()
            self.h_action = self.action.copy()
            self._policy_reasoning()
            # print(self.motion.joint_pos[self.time_step],"\r\n",self.r_joint_pos)
            action = (
                np.clip(
                    copy.deepcopy(self.action[self.isaac_sim2mujoco_index]),
                    -self.action_clip,
                    self.action_clip,
                )
                * self.action_scale
                * self.tq_max
                / self.P_gains
                + self.default_pos
            )
            target_q = action.clip(-self.action_clip, self.action_clip)
            # print(target_q)
            self.target_dof_pos = target_q  # + self.default_pos[: self.action_num]
        self.time_step += 1
        self.contact_force()
        self.sim_loop()

        # mujoco.mjr_render(self._viewport, self._scene, self._context)
        # im = self.read_pixels()
        # self.video_recorder(im)
        # 更新 Renderer 场景，使用查看器的相机和选项，使图像与窗口一致
        self.renderer.update_scene(
            self.d,
            camera=self.viewer.cam,  # 使用查看器的相机视图
            scene_option=self.viewer.opt,  # 使用查看器的渲染选项
        )

        # 捕获图像：返回 (height, width, 3) 的 uint8 NumPy 数组 (RGB)
        img = self.renderer.render()
        self.video_recorder(img)

        self.viewer.sync()
        self.update_vel_geom()

    def update_obs(self, time_step):
        """
        +----------------------------------------------------------+
        | Active Observation Terms in Group: 'policy' (shape: (150,)) |
        +------------+--------------------------------+------------+
        |   Index    | Name                           |   Shape    |
        +------------+--------------------------------+------------+
        |     0      | command                        |   (54,)    |
        |     1      | motion_ref_pos_b               |    (3,)    |
        |     2      | motion_ref_ori_b               |    (6,)    |
        |     3      | base_lin_vel                   |    (3,)    |
        |     4      | base_ang_vel                   |    (3,)    |
        |     5      | joint_pos                      |   (27,)    |
        |     6      | joint_vel                      |   (27,)    |
        |     7      | actions                        |   (27,)    |
        +------------+--------------------------------+------------+
        """
        #################
        # motion joint pos command 27
        #################
        self.single_obs[:27] = self.motion.joint_pos[time_step]
        #################
        # motion joint vel command 27
        #################
        self.single_obs[27:54] = self.motion.joint_vel[time_step]

        body_name = "torso_link"  # robot_ref_body_index=3 motion_ref_body_index=7
        # body_name = "pelvis"
        body_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            raise ValueError(f"Body {body_name} not found in model")
        position = self.d.xpos[body_id]
        quaternion = self.d.xquat[body_id]
        self.robot_ref_pos_w = torch.from_numpy(position).unsqueeze(0)  # shape [n,3]
        self.robot_ref_quat_w = torch.from_numpy(quaternion).unsqueeze(0)  # shape [n,4]
        self.ref_pos_w = self.motion.body_pos_w[time_step, 7, :]  # shape [n,3]
        # self.ref_pos_w[:, 0] += 1.25
        self.ref_quat_w = self.motion.body_quat_w[time_step, 7, :]  # shape [n,4]
        pos, ori = subtract_frame_transforms(
            self.robot_ref_pos_w, self.robot_ref_quat_w, self.ref_pos_w, self.ref_quat_w
        )  # shape [n,3][n,4]
        mat = matrix_from_quat(ori)
        motion_ref_pos_b = pos.view(1, -1)  # shape [n,3]
        motion_ref_ori_b = mat[..., :2].reshape(mat.shape[0], -1)  # shape [n,6]
        #################
        # motion_ref_pos_b 3
        #################
        self.single_obs[54:57] = motion_ref_pos_b
        #################
        # motion_ref_ori_b 6
        #################
        self.single_obs[57:63] = motion_ref_ori_b
        #################
        # base_lin_vel 3
        #################
        self.single_obs[63:66] = (
            quat_rotate_inverse(
                torch.from_numpy(self.d.qpos[3:7]).unsqueeze(0),
                torch.from_numpy(self.d.qvel[0:3]).unsqueeze(0),
            )
            .detach()
            .cpu()
            .numpy()
        )
        #################
        # base_ang_vel 3
        #################
        self.single_obs[66:69] = (
            quat_rotate_inverse(
                torch.from_numpy(self.d.qpos[3:7]).unsqueeze(0),
                torch.from_numpy(self.d.qvel[3:6]).unsqueeze(0),
            )
            .detach()
            .cpu()
            .numpy()
        )
        #################
        # joint_pos 27
        #################
        self.single_obs[69:96] = (
            self.d.qpos[7:] - self.default_pos[: self.action_num]
        )[self.mujoco2isaac_sim_index]
        #################
        # joint_vel 27
        #################
        self.single_obs[96:123] = self.d.qvel[6:][self.mujoco2isaac_sim_index]
        #################
        # actions 27
        #################
        self.single_obs[123:150] = self.action  # / self.action_scale

        self.obs = (
            torch.tensor(np.concatenate([self.single_obs] * cfg.frame_stack, axis=-1))
            .clamp(-10, 10)
            .unsqueeze(0)
            .detach()
            .cpu()
            .numpy()
        )

    def _policy_reasoning(self):

        if cfg.policy_type == "onnx":
            (
                act,
                self.r_joint_pos,
                self.r_joint_vel,
                self.r_body_pos_w,
                self.r_body_quat_w,
                self.r_body_lin_vel_w,
                self.r_body_ang_vel_w,
            ) = self.run_onnx_inference(
                self.policy, self.obs.astype(np.float32), self.time_step
            )
        self.action[:] = act.copy()

    def sim_loop(self):
        for i in range(self.control_decimation):
            step_start = time.time()

            if not cfg.motion_play or (cfg.motion_play and cfg.sim_motion_play):
                tau = self._PD_control(self.target_dof_pos)
                self.d.ctrl[:] = tau
            if not self.paused:
                self.prev_qpos = self.d.qpos.copy()
                self.set_camera()

                mujoco.mj_step(self.m, self.d)

            time_until_next_step = self.m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    def _PD_control(self, _P_t=0):
        P_n = self.d.qpos[7:]
        V_n = self.d.qvel[6:]

        KP = self.P_gains
        KD = self.D_gains
        # 在_compute_torques中使用
        t = KP * (_P_t - P_n) - KD * V_n
        return t

    def contact_force(self):
        force = 0
        for contact_id, contact in enumerate(self.d.contact):
            if contact.efc_address >= 0:  # Valid contact
                forcetorque = np.zeros(6)
                mujoco.mj_contactForce(self.m, self.d, contact_id, forcetorque)
                # print("forcetorque: ",forcetorque)
                force += forcetorque[0]
        self.fz = force / 65 / 9.81
        # print("force: %8.3f"% force)

    def key_callback(self, keycode):
        # 按空格键切换暂停/继续

        if chr(keycode) == " ":
            self.paused = not self.paused
            print(f"Simulation {'paused' if self.paused else 'running'}")
        elif chr(keycode).lower() == "w":
            self.cmd[1] = 0.0
            self.cmd[2] = 0.0
            self.cmd[0] = 0.8
        elif chr(keycode).lower() == "s":
            self.cmd[0] = -0.8
            self.cmd[1] = 0.0
            self.cmd[2] = 0.0

        elif chr(keycode).lower() == "a":
            self.cmd[1] = 0.4
            self.cmd[0] = 0.0
            self.cmd[2] = 0.0
        elif chr(keycode).lower() == "d":
            self.cmd[1] = -0.4
            self.cmd[0] = 0.0
            self.cmd[2] = 0.0
        elif chr(keycode).lower() == "q":
            self.cmd[2] = 1.5
            self.cmd[0] = 0.0
            self.cmd[1] = 0.0
        elif chr(keycode).lower() == "e":
            self.cmd[2] = -1.5
            self.cmd[0] = 0.0
            self.cmd[1] = 0.0
        # 释放键时重置控制量
        elif keycode == 48:  # keycode=0 表示无按键
            self.cmd[0] = 0.0
            self.cmd[1] = 0.0
            self.cmd[2] = 0.0

    def set_camera(self):
        self.viewer.cam.distance = 4
        self.viewer.cam.azimuth = 135
        self.viewer.cam.elevation = 0.0
        self.viewer.cam.fixedcamid = -1
        self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.viewer.cam.trackbodyid = 0

    def _init_robot_conf(self):
        self.default_pos = np.array(
            cfg.leg_default_pos + cfg.torso_default_pos + cfg.arm_default_pos,
            dtype=np.float32,
        )
        self.P_gains = np.array(
            cfg.leg_P_gains + cfg.torso_P_gains + cfg.arm_P_gains, dtype=np.float32
        )
        self.D_gains = np.array(
            cfg.leg_D_gains + cfg.torso_D_gains + cfg.arm_D_gains, dtype=np.float32
        )
        if cfg.only_leg_flag:
            self.default_pos = self.default_pos[:12]
            self.P_gains = self.P_gains[:12]
            self.D_gains = self.D_gains[:12]
        else:
            if not cfg.with_wrist_flag:
                indices = [
                    16,
                    17,
                    18,
                    19,
                    23,
                    24,
                    25,
                    26,
                ]
                self.default_pos = np.delete(self.default_pos, indices)
                self.P_gains = np.delete(self.P_gains, indices)
                self.D_gains = np.delete(self.D_gains, indices)
        self.P_n = np.zeros_like(self.default_pos)
        self.V_n = np.zeros_like(self.default_pos)
        self.cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.isaac_sim2mujoco_index = [
            cfg.isaac_sim_joint_name.index(name) for name in cfg.mujoco_joint_name
        ]
        self.mujoco2isaac_sim_index = [
            cfg.mujoco_joint_name.index(name) for name in cfg.isaac_sim_joint_name
        ]

    def _init_policy_conf(self):
        """
        ['pelvis', 0
        'left_hip_roll_link', 1
        'left_knee_link', 2
        'left_ankle_roll_link', 3
        'right_hip_roll_link', 4
        'right_knee_link', 5
        'right_ankle_roll_link', 6
        'torso_link', 7
        'left_shoulder_roll_link', 8
        'left_elbow_link', 9
        'left_wrist_yaw_link', 10
        'right_shoulder_roll_link', 11
        'right_elbow_link',12
        'right_wrist_yaw_link']13
        """
        self.body_indexes = torch.tensor(
            [0, 8, 12, 20, 9, 13, 21, 3, 10, 18, 26, 11, 19, 27],
            dtype=torch.long,
            device="cpu",
        )
        self.motion = MotionLoader(
            cfg.motion_file,
            self.body_indexes,
            "cpu",
        )
        self.policy_dt = cfg.policy_dt
        if cfg.motion_play:
            self.policy_dt = 1 / self.motion.fps
        self.control_decimation = int(self.policy_dt / cfg.simulator_dt)
        print("control_decimation: ", self.control_decimation)
        if cfg.policy_type == "torch":
            self.policy = torch.jit.load(cfg.policy_path)
        elif cfg.policy_type == "onnx":
            self.policy = self.load_onnx_model(cfg.policy_path)

        self.h2_action = np.zeros(cfg.action_num, dtype=np.float32)
        self.h_action = np.zeros(cfg.action_num, dtype=np.float32)
        self.action = np.zeros(cfg.action_num, dtype=np.float32)
        self.action_clip = cfg.action_clip

        self.tq_max = np.array(
            cfg.leg_tq_max + cfg.torso_tq_max + cfg.arm_tq_max,
            dtype=np.float32,
        )
        self.action_scale = cfg.action_scale
        self.action_num = cfg.action_num
        self.obs = np.zeros(cfg.num_single_obs * cfg.frame_stack, dtype=np.float32)
        self.time_step = np.zeros(1, dtype=np.float32)

        self.single_obs = np.zeros(cfg.num_single_obs, dtype=np.float32)

    def load_onnx_model(self, onnx_path, device="cpu"):
        providers = (
            ["CPUExecutionProvider"] if device == "cpu" else ["CUDAExecutionProvider"]
        )
        session = ort.InferenceSession(onnx_path, providers=providers)
        return session

    def run_onnx_inference(self, session, obs, time_step):
        # 转换为numpy array并确保数据类型正确
        if isinstance(obs, torch.Tensor):
            obs = obs.detach().cpu().numpy()
        if isinstance(time_step, torch.Tensor):
            time_step = time_step.detach().cpu().numpy()
        # 获取输入名称
        obs_name = session.get_inputs()[0].name
        time_step_name = session.get_inputs()[1].name

        # 运行推理
        (
            actions,
            joint_pos,
            joint_vel,
            body_pos_w,
            body_quat_w,
            body_lin_vel_w,
            body_ang_vel_w,
        ) = session.run(
            None,
            {
                obs_name: obs.reshape(1, 150),
                time_step_name: time_step.reshape(1, 1),
            },
        )
        return (
            actions,
            joint_pos,
            joint_vel,
            body_pos_w,
            body_quat_w,
            body_lin_vel_w,
            body_ang_vel_w,
        )  # 默认返回第一个输出

    def _rehandle_xml(self):

        joints_to_remove, actuators_to_remove, _ = self._get_spec_modifications(
            only_leg=cfg.only_leg_flag, with_wrist=cfg.with_wrist_flag
        )
        for actuator in self.spec.actuators:
            if actuator.name in actuators_to_remove:
                actuator.delete()
        for joint in self.spec.joints:
            if joint.name in joints_to_remove:
                joint.delete()

    def _get_spec_modifications(
        self, only_leg, with_wrist
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Specifies which joints, actuators, and equality constraints should be removed from the Mujoco specification.

        Returns:
            Tuple[List[str], List[str], List[str]]: A tuple containing lists of joints to remove, actuators to remove,
            and equality constraints to remove.
        """

        joints_to_remove = [
            # Left Hand
            "L_thumb_proximal_yaw_joint",
            "L_thumb_proximal_pitch_joint",
            "L_thumb_intermediate_joint",
            "L_thumb_distal_joint",
            "L_index_proximal_joint",
            "L_index_intermediate_joint",
            "L_middle_proximal_joint",
            "L_middle_intermediate_joint",
            "L_ring_proximal_joint",
            "L_ring_intermediate_joint",
            "L_pinky_proximal_joint",
            "L_pinky_intermediate_joint",
            # Right Hand
            "R_thumb_proximal_yaw_joint",
            "R_thumb_proximal_pitch_joint",
            "R_thumb_intermediate_joint",
            "R_thumb_distal_joint",
            "R_index_proximal_joint",
            "R_index_intermediate_joint",
            "R_middle_proximal_joint",
            "R_middle_intermediate_joint",
            "R_ring_proximal_joint",
            "R_ring_intermediate_joint",
            "R_pinky_proximal_joint",
            "R_pinky_intermediate_joint",
        ]

        actuators_to_remove = [
            # Left Hand
            "L_thumb_proximal_yaw_joint",
            "L_thumb_proximal_pitch_joint",
            "L_thumb_intermediate_joint",
            "L_thumb_distal_joint",
            "L_index_proximal_joint",
            "L_index_intermediate_joint",
            "L_middle_proximal_joint",
            "L_middle_intermediate_joint",
            "L_ring_proximal_joint",
            "L_ring_intermediate_joint",
            "L_pinky_proximal_joint",
            "L_pinky_intermediate_joint",
            # Right Hand
            "R_thumb_proximal_yaw_joint",
            "R_thumb_proximal_pitch_joint",
            "R_thumb_intermediate_joint",
            "R_thumb_distal_joint",
            "R_index_proximal_joint",
            "R_index_intermediate_joint",
            "R_middle_proximal_joint",
            "R_middle_intermediate_joint",
            "R_ring_proximal_joint",
            "R_ring_intermediate_joint",
            "R_pinky_proximal_joint",
            "R_pinky_intermediate_joint",
        ]
        if not with_wrist:
            joints_to_remove += [
                "left_wrist_roll_joint",
                "left_wrist_pitch_joint",
                "left_wrist_yaw_joint",
                "right_wrist_roll_joint",
                "right_wrist_pitch_joint",
                "right_wrist_yaw_joint",
            ]
            actuators_to_remove += [
                "left_wrist_roll_joint",
                "left_wrist_pitch_joint",
                "left_wrist_yaw_joint",
                "right_wrist_roll_joint",
                "right_wrist_pitch_joint",
                "right_wrist_yaw_joint",
            ]
        if only_leg:
            joints_to_remove += [
                # Left Arm
                "left_shoulder_pitch_joint",
                "left_shoulder_roll_joint",
                "left_shoulder_yaw_joint",
                "left_elbow_joint",
                # Right Arm
                "right_shoulder_pitch_joint",
                "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint",
                "right_elbow_joint",
                "torso_joint",
            ]
            actuators_to_remove += [
                # Left Arm
                "left_shoulder_pitch_joint",
                "left_shoulder_roll_joint",
                "left_shoulder_yaw_joint",
                "left_elbow_joint",
                # Right Arm
                "right_shoulder_pitch_joint",
                "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint",
                "right_elbow_joint",
                "torso_joint",
            ]

        equ_constr_to_remove = []

        return joints_to_remove, actuators_to_remove, equ_constr_to_remove

    def plot_data(self, data_queue):
        print("plot_data")
        plt.ion()  # 开启交互模式
        first_flag = 1

        while True:
            if not data_queue.empty():
                merged_tensor = data_queue.get()
                plot_num = merged_tensor.shape[0]
                if first_flag:
                    first_flag = 0
                    # 计算行数和列数
                    rows = math.floor(math.sqrt(plot_num))
                    cols = math.ceil(plot_num / rows)

                    fig, axs = plt.subplots(rows, cols, figsize=(10, 12))  # 创建子图
                    axs = axs.flatten()  # 将二维数组展平成一维数组，方便索引

                    lines = [ax.plot([], [])[0] for ax in axs]  # 初始化每个子图的线条
                    xdata = [
                        [0 for _ in range(700)] for _ in range(plot_num)
                    ]  # 存储每个子图的 x 数据
                    ydata = [
                        [0] * 700 for _ in range(plot_num)
                    ]  # 存储每个子图的 y 数据

                    from matplotlib.widgets import Slider

                    # Add slider
                    ax_slider = plt.axes([0.15, 0.02, 0.65, 0.03])  # Slider position
                    self.slider = Slider(
                        ax_slider, "Control", 0.1, 3.0, valinit=1.0, valstep=0.001
                    )
                    self.slider.on_changed(self.update_sld)
                for i in range(plot_num):
                    xdata[i].append(len(xdata[i]))
                    ydata[i].append(merged_tensor[i].item())
                    lines[i].set_data(xdata[i][-100:], ydata[i][-100:])
                    axs[i].relim()
                    axs[i].autoscale_view()
                # print(len(xdata[i]))
                if len(xdata[i]) % 1 == 0:
                    fig.canvas.draw()
                    fig.canvas.flush_events()

    def update_sld(self, val):
        slider_value = self.slider.val  # Get slider value
        self.D_gains[self.change_id] = slider_value * self.D_gains[self.change_id + 6]
        print(f"D_gains {self.change_id:d} value: {self.D_gains[self.change_id]:.2f}")

    def init_vel_geom(self, input):
        # create an invisibale geom and add label on it
        geom = self.viewer.user_scn.geoms[self.viewer.user_scn.ngeom]
        mujoco.mjv_initGeom(
            geom,
            type=mujoco.mjtGeom.mjGEOM_LABEL,
            size=np.array([0.2, 0.2, 0.2]),  # label_size
            pos=self.d.qpos[:3]
            + np.array(
                [0.0, 0.0, 1.0]
            ),  # lebel position, here is 1 meter above the root joint
            mat=np.eye(3).flatten(),  # label orientation, here is no rotation
            rgba=np.array([0, 0, 0, 0]),  # invisible
        )
        geom.label = str(input)  # set label text
        self.viewer.user_scn.ngeom += 1

    def update_vel_geom(self):
        # update the geom position and label text
        geom = self.viewer.user_scn.geoms[self.viewer.user_scn.ngeom - 1]
        geom.pos = self.d.qpos[:3] + np.array([0.0, 0.0, 1.0])
        geom.label = "rb h{:.2f} \r\nGoal Vel: x: {:.2f}, y: {:.2f}, yaw: {:.2f},force_z: {:.2f}".format(
            # self.data["robot.data.body_pos_w"].detach().cpu().numpy()[0][2],
            0.0,
            self.cmd[0],
            self.cmd[1],
            self.cmd[2],
            self.fz,
        )


if __name__ == "__main__":
    s = simulator()
    s.run()
