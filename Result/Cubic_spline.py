import numpy as np
import pandas as pd


def normalize(q):
    """Нормализует кватернион."""
    return q / np.linalg.norm(q)


def quaternion_dot(q1, q2):
    """Скалярное произведение кватернионов."""
    return np.dot(q1, q2)


def slerp_linear(q0, q1, t):
    """Упрощённый SLERP: линейная интерполяция + нормализация."""
    q0 = normalize(q0)
    q1 = normalize(q1)

    dot = quaternion_dot(q0, q1)
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    result = (1 - t) * q0 + t * q1
    return normalize(result)


class NaturalCubicSpline1D:
    """Одномерный кубический сплайн."""

    def __init__(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        n = len(x)
        if n < 2:
            raise ValueError("Должно быть минимум 2 узла")
        if np.any(np.diff(x) <= 0):
            raise ValueError("x должен строго возрастать")

        h = np.diff(x)
        alpha = np.zeros(n)
        for i in range(1, n - 1):
            alpha[i] = (3 / h[i]) * (y[i + 1] - y[i]) - \
                (3 / h[i - 1]) * (y[i] - y[i - 1])

        l = np.ones(n)
        mu = np.zeros(n)
        z = np.zeros(n)

        for i in range(1, n - 1):
            l[i] = 2 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1]
            mu[i] = h[i] / l[i]
            z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]

        b = np.zeros(n - 1)
        c = np.zeros(n)
        d = np.zeros(n - 1)

        for j in reversed(range(n - 1)):
            c[j] = z[j] - mu[j] * c[j + 1]
            b[j] = ((y[j + 1] - y[j]) / h[j]) - \
                (h[j] * (c[j + 1] + 2 * c[j]) / 3)
            d[j] = (c[j + 1] - c[j]) / (3 * h[j])

        self.x = x
        self.y = y
        self.b = b
        self.c = c[:-1]
        self.d = d
        self.h = h

    def __call__(self, t):
        # Ограничиваем t в диапазоне узлов
        t = np.clip(t, self.x[0], self.x[-1])
        # Найти индекс интервала
        i = np.searchsorted(self.x, t) - 1
        if i < 0:
            i = 0
        elif i >= len(self.b):
            i = len(self.b) - 1
        dx = t - self.x[i]
        return self.y[i] + self.b[i] * dx + self.c[i] * dx ** 2 + self.d[i] * dx ** 3


class RigidBodyTrajectory:
    """Интерполяция позиции кубическим сплайном и ориентации упрощённым SLERP."""

    def __init__(self, time_list, position_list, orientation_list):
        # Сортируем по времени
        idx = np.argsort(time_list)
        self.times = np.array(time_list)[idx]
        self.positions = np.array(position_list)[idx]
        self.orientations = np.array(orientation_list)[idx]

        # Создаем сплайны для x, y, z
        self.spline_x = NaturalCubicSpline1D(self.times, self.positions[:, 0])
        self.spline_y = NaturalCubicSpline1D(self.times, self.positions[:, 1])
        self.spline_z = NaturalCubicSpline1D(self.times, self.positions[:, 2])

    def interpolate(self, t):
        # Ограничение времени
        if t <= self.times[0]:
            pos = self.positions[0]
            quat = normalize(self.orientations[0])
            return pos, quat
        if t >= self.times[-1]:
            pos = self.positions[-1]
            quat = normalize(self.orientations[-1])
            return pos, quat

        # Найти индекс интервала
        i = np.searchsorted(self.times, t) - 1
        t0, t1 = self.times[i], self.times[i + 1]
        alpha = (t - t0) / (t1 - t0)

        # Интерполировать позицию кубическим сплайном
        pos = np.array([
            self.spline_x(t),
            self.spline_y(t),
            self.spline_z(t)
        ])

        # Интерполировать ориентацию упрощённым SLERP
        q0 = self.orientations[i]
        q1 = self.orientations[i + 1]
        quat = slerp_linear(q0, q1, alpha)

        return pos, quat

    def sample(self, t0=None, t1=None, step=0.1):
        if t0 is None:
            t0 = self.times[0]
        if t1 is None:
            t1 = self.times[-1]

        times = np.arange(t0, t1 + 1e-8, step)
        records = []
        for t in times:
            pos, quat = self.interpolate(t)
            records.append({'time': t, 'position': pos, 'orientation': quat})
        return pd.DataFrame(records)


if __name__ == "__main__":
    times = [0.0, 0.7, 1.5, 2.0]
    poses = [
        [0, 0, 0],
        [1, 2, 0],
        [0, 2, -1],
        [3, 1, 2]
    ]
    quats = [
        [0, 0, 0, 1],
        [0, 0.382683, 0, 0.923880],
        [0, 0, 1, 0],
        [0, 1, 0, 0]
    ]

    traj = RigidBodyTrajectory(times, poses, quats)
    df = traj.sample(step=0.2)
    print(df)
