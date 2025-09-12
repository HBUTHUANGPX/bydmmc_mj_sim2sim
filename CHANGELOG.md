
2025.9.12
-
1. 补充了asset中的H1_2模型的mesh文件。
2. 关于 deploy_mujoco.py 的修改：
    - 修改了关于 `video_recoder` 模块的引用拼写错误
    - 修改了 `cfg` 中的相关的路径定义
    - 修改了obs的维度,减少了 `motion_ref_pos_b` 和 `base_lin_vel`
    - 使用 `Pinocchio` 库构建并新增了`pin_mj`类，修改替换了`motion_ref_ori_b` 的计算流程，不再使用mujoco的xquat获得数据，而是使用 `Pinocchio` 库进行运动学计算。
    - 训练的数据集修改为本人使用xsens录制的行走数据。