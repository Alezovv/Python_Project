import numpy as np
import pandas as pd


def normalize(q):
    """Функция нормализует кватернион, так как только единичные кватернионы представляют допустимые вращения:"""
    return q / np.linalg.norm(q)


def quaternion_dot(q1, q2):
    """Скалярное произведение кватернионов."""
    return np.dot(q1, q2)


def slerp_linear(q0, q1, t):
    """Упрощённый SLERP"""
    q0 = normalize(q0)
    q1 = normalize(q1)

    dot = quaternion_dot(q0, q1)
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    result = (1 - t) * q0 + t * q1
    return normalize(result)


class RigidBodyLinear:
    def __init__(self, time_list, position_list, orientation_list):
        if len(time_list) < 2:
            raise ValueError("Нужно как минимум две точки")

        self.times = np.array(time_list)
        self.positions = np.array(position_list)
        self.orientations = np.array(orientation_list)

        # Проверка
        if not np.all(np.diff(self.times) > 0):
            raise ValueError("Временные метки должны строго возрастать")

    def interpolate(self, t):
        # Обработка выходов за границы
        if t <= self.times[0]:
            return self.positions[0], normalize(self.orientations[0])
        if t >= self.times[-1]:
            return self.positions[-1], normalize(self.orientations[-1])

        i = np.searchsorted(self.times, t) - 1
        t0, t1 = self.times[i], self.times[i + 1]
        alpha = (t - t0) / (t1 - t0)

        p0, p1 = self.positions[i], self.positions[i + 1]
        pos = (1 - alpha) * p0 + alpha * p1

        q0, q1 = self.orientations[i], self.orientations[i + 1]
        quat = slerp_linear(q0, q1, alpha)

        return pos, quat

    def write(self, step=0.1, t0=None, t1=None):
        if t0 is None:
            t0 = self.times[0]
        if t1 is None:
            t1 = self.times[-1]

        times = np.arange(t0, t1 + 1e-8, step)
        data = []
        for t in times:
            pos, quat = self.interpolate(t)
            data.append({'time': t, 'position': pos, 'orientation': quat})
        return pd.DataFrame(data)


if __name__ == "__main__":
    times = [0.0, 0.5, 1.0, 1.5]
    poses = [
        [0, 0, 0],
        [1, 1, 0],
        [2, 0, -1],
        [3, 0, 0]
    ]
    quats = [
        [0, 0, 0, 1],
        [0, 0, 0.7071, 0.7071],
        [0, 0, 1, 0],
        [0, 0, -0.7071, 0.7071]
    ]

    traj = RigidBodyLinear(times, poses, quats)
    df = traj.write(step=0.2)
    print(df)
