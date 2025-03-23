# import os
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation, writers
# import numpy as np
#
# #from vpython import *
# from datetime import datetime
#
# def render_animation(skeleton, pose, azim=0.0, bitrate=3000, output=None):
#
#     # t_total = pose.shape[0]
#     radius = 1.7
#     lines_3d = []
#
#     plt.ioff()
#     fig = plt.figure(figsize = (6,6))
#     fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
#     ax = fig.add_subplot(111, projection='3d')
#     ax.view_init(elev=15, azim=azim)
#     ax.set_xlim3d([-radius / 2, radius / 2])
#     ax.set_ylim3d([-radius / 2, radius / 2])
#     ax.set_zlim3d([0, radius])
#     # ax.set_xticklabels([])
#     # ax.set_yticklabels([])
#     # ax.set_zticklabels([])
#     ax.dist = 5.0
#     # ax.set_axis_off()
#     ax.patch.set_alpha(0.0)
#
#     ax_3d = ax
#     trajectories = pose[:, 0, [0, 1]]
#
#     # fig.tight_layout(h_pad=15, w_pad=15)
#     # fig.subplots_adjust(wspace=-0.4, hspace=0.4)
#
#     anim = None
#     initialized = False
#     animating = True
#     find = 0
#     hist_lcol, hist_mcol, hist_rcol = 'blue', 'black', 'red'
#     # pred_lcol, pred_mcol, pred_rcol = 'purple', 'black', 'green'
#
#     parents = skeleton.parents()
#
#     def update_video(i):
#         nonlocal initialized
#         lcol, mcol, rcol = hist_lcol, hist_mcol, hist_rcol
#
#         trajectories = pose[:, 0, [0, 1, 2]]
#         ax.set_xlim3d([-radius / 2 + trajectories[i, 0], radius / 2 + trajectories[i, 0]])
#         ax.set_ylim3d([-radius / 2 + trajectories[i, 1], radius / 2 + trajectories[i, 1]])
#         ax.set_zlim3d([-radius / 2 + trajectories[i, 2], radius / 2 + trajectories[i, 2]])
#
#         if not initialized:
#             for j, j_parents in enumerate(parents):
#                 if j_parents == -1:
#                     continue
#                 if j in skeleton.joints_right():
#                     col = rcol
#                 elif j in skeleton.joints_left():
#                     col = lcol
#                 else:
#                     col = mcol
#
#                 pos = pose[i]
#
#                 lines_3d.append(ax.plot([pos[j, 0], pos[j_parents, 0]],
#                                         [pos[j, 1], pos[j_parents, 1]],
#                                         [pos[j, 2], pos[j_parents, 2]], zdir='z', c=col, linewidth=3.0))
#
#                 # # Find minimum and maximum values across all frames
#                 # min_x, max_x = np.min(pose[:, :, 0]), np.max(pose[:, :, 0])
#                 # min_y, max_y = np.min(pose[:, :, 1]), np.max(pose[:, :, 1])
#                 # min_z, max_z = np.min(pose[:, :, 2]), np.max(pose[:, :, 2])
#                 #
#                 # # Set axis limits with a buffer of radius/2
#                 # ax.set_xlim3d(-radius / 2 + min_x, radius / 2 + max_x)
#                 # ax.set_ylim3d(-radius / 2 + min_y, radius / 2 + max_y)
#                 # ax.set_zlim3d(-radius / 2 + min_z, radius / 2 + max_z)
#
#             initialized = True
#
#         else:
#             for j, j_parents in enumerate(parents):
#                 if j_parents == -1:
#                     continue
#
#                 if j in skeleton.joints_right():
#                     col = rcol
#                 elif j in skeleton.joints_left():
#                     col = lcol
#                 else:
#                     col = mcol
#
#
#                 pos = pose[i]
#                 x_array = np.array([pos[j, 0], pos[j_parents, 0]])
#                 y_array = np.array([pos[j, 1], pos[j_parents, 1]])
#                 z_array = np.array([pos[j, 2], pos[j_parents, 2]])
#                 lines_3d[j - 1][0].set_data_3d(x_array, y_array, z_array)
#                 lines_3d[j - 1][0].set_color(col)
#
#         # if i % 49 == 0:
#         #     pos = pose[i]
#         #     print(pos)
#
#     def show_animation():
#         nonlocal anim
#         if anim is not None:
#             anim.event_source.stop()
#         anim = FuncAnimation(fig, update_video, frames=np.arange(0, pose.shape[0]), interval=0, repeat=True)
#         plt.draw()
#
#     def save():
#         nonlocal anim
#
#         fps=50
#         anim = FuncAnimation(fig, update_video, frames=np.arange(0, pose.shape[0]), interval=1000 / fps, repeat=False)
#         os.makedirs(os.path.dirname(output), exist_ok=True)
#         if output.endswith('.mp4'):
#             Writer = writers['ffmpeg']
#             writer = Writer(fps=fps, metadata={}, bitrate=bitrate)
#             anim.save(output, writer=writer)
#         elif output.endswith('.gif'):
#             anim.save(output, dpi=80, writer='pillow')
#         else:
#             raise ValueError('Unsupported output format (only .mp4 and .gif are supported)')
#         print(f'Video saved to {output}!')
#
#     save()
#     show_animation()
#     plt.show()
#     plt.close()
#     a = ['S1', 'Walking', 350, 17, (0.9, 0.89, 0.91)]

import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
import numpy as np


def render_animation(skeleton, pose, azim=0.0, bitrate=3000, output=None):
    # 设定动画参数
    radius = 1.7

    # 计算固定的参考点（如腰部），这里选择 index 0, 3, 6 作为参考
    fixed_nodes = [0, 3, 6]
    reference_point = np.mean(pose[0, fixed_nodes, :], axis=0)  # 计算第一帧的固定中心
    adjusted_pose = pose - reference_point  # 让所有帧都相对于这个中心调整

    # 计算整个数据的最小和最大范围，避免动态变化影响观看体验
    min_xyz = np.min(adjusted_pose, axis=(0, 1))
    max_xyz = np.max(adjusted_pose, axis=(0, 1))
    center_xyz = (min_xyz + max_xyz) / 2
    range_xyz = max_xyz - min_xyz
    axis_limit = np.max(range_xyz) / 2

    # 设定可视化
    plt.ioff()
    fig = plt.figure(figsize=(6, 6))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax = fig.add_subplot(111, projection='3d')

    ax.view_init(elev=15, azim=azim)
    ax.set_xlim3d([center_xyz[0] - axis_limit, center_xyz[0] + axis_limit])
    ax.set_ylim3d([center_xyz[1] - axis_limit, center_xyz[1] + axis_limit])
    ax.set_zlim3d([center_xyz[2] - axis_limit, center_xyz[2] + axis_limit])
    ax.dist = 5.0
    ax.patch.set_alpha(0.0)

    # 设定颜色
    hist_lcol, hist_mcol, hist_rcol = 'purple', 'green', 'green'
    parents = skeleton.parents()

    def update_video(i):
        ax.clear()  # **关键：清除旧的绘图，防止静止骨架残留**

        # 重新设定相机视角（因为 ax.clear() 会重置所有参数）
        ax.view_init(elev=15, azim=azim)
        ax.set_xlim3d([center_xyz[0] - axis_limit, center_xyz[0] + axis_limit])
        ax.set_ylim3d([center_xyz[1] - axis_limit, center_xyz[1] + axis_limit])
        ax.set_zlim3d([center_xyz[2] - axis_limit, center_xyz[2] + axis_limit])
        ax.dist = 5.0

        pos = adjusted_pose[i]

        # 画动态骨架
        for j, j_parent in enumerate(parents):
            if j_parent == -1:
                continue
            col = hist_rcol if j in skeleton.joints_right() else (
                hist_lcol if j in skeleton.joints_left() else hist_mcol)
            ax.plot([pos[j, 0], pos[j_parent, 0]],
                    [pos[j, 1], pos[j_parent, 1]],
                    [pos[j, 2], pos[j_parent, 2]], zdir='z', c=col, linewidth=3.0)

    def show_animation():
        anim = FuncAnimation(fig, update_video, frames=np.arange(0, adjusted_pose.shape[0]), interval=30, repeat=True)
        plt.show()

    def save():
        fps = 50
        anim = FuncAnimation(fig, update_video, frames=np.arange(0, adjusted_pose.shape[0]), interval=1000 / fps,
                             repeat=False)
        os.makedirs(os.path.dirname(output), exist_ok=True)
        if output.endswith('.mp4'):
            Writer = writers['ffmpeg']
            writer = Writer(fps=fps, metadata={}, bitrate=bitrate)
            anim.save(output, writer=writer)
        elif output.endswith('.gif'):
            anim.save(output, dpi=80, writer='pillow')
        else:
            raise ValueError('Unsupported output format (only .mp4 and .gif are supported)')
        print(f'Video saved to {output}!')

    save()
    show_animation()
    plt.close()
