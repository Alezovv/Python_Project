import Slerp_func
import Vizual
import Cubic_spline

import pandas as pd

# --- Линейная интерполяция по 4 точкам ---
times_linear = [0.0, 0.5, 1.0, 1.5]
poses_linear = [
    [0, 0, 0],
    [1, 1, 0],
    [2, 0, -1],
    [3, 0, 0]
]
quats_linear = [
    [0, 0, 0, 1],
    [0, 0, 0.7071, 0.7071],
    [0, 0, 1, 0],
    [0, 0, -0.7071, 0.7071]
]

traj_linear = Slerp_func.RigidBodyLinear(
    times_linear, poses_linear, quats_linear)
df_linear = traj_linear.write(step=0.2)

# Сохраняем позиции и ориентации в отдельные файлы
df_linear_orientations = df_linear[['time']].copy()
df_linear_orientations[['qx', 'qy', 'qz', 'qw']] = pd.DataFrame(
    df_linear['orientation'].tolist(), index=df_linear.index)
df_linear_orientations.to_csv("linear_orientations.csv", index=False)

# Визуализация
viz_linear = Vizual.RigidBodyVisualizer(df_linear)
viz_linear.animate()

# --- Сплайн-интерполяция ---
times_spline = [0.0, 0.7, 1.5, 3.0, 3.7, 4.2]
poses_spline = [
    [0, 0, 0],
    [1, 2, 0],
    [0, 2, -1],
    [-1, 0, 0],
    [2, 1, 2],
    [3, 0, 0]
]
quats_spline = [
    [0, 0, 0, 1],
    [0, 0.382683, 0, 0.923880],
    [0, 0, 1, 0],
    [0, -0.707107, 0, 0.707107],
    [1, 0, 0, 0],
    [0, 0, 0.707107, 0.707107]
]

traj_spline = Cubic_spline.RigidBodyTrajectory(
    times_spline, poses_spline, quats_spline)
df_spline = traj_spline.sample(step=0.2)

# Сохраняем данные траектории со сплайнами
df_spline_positions = df_spline[['time']].copy()
df_spline_positions[['x', 'y', 'z']] = pd.DataFrame(
    df_spline['position'].tolist(), index=df_spline.index)
df_spline_positions.to_csv("spline_positions.csv", index=False)

# Визуализация
viz_spline = Vizual.RigidBodyVisualizer(df_spline)
viz_spline.animate()
