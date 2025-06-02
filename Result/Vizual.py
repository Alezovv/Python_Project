import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class RigidBodyVisualizer:
    def __init__(self, df, axis_length=0.3, max_fps=10, min_fps=2):
        self.df = df.sort_values('time').reset_index(drop=True)
        self.axis_length = axis_length
        self.max_fps = max_fps
        self.min_fps = min_fps

    @staticmethod
    def normalize_quat(q):
        q = np.array(q, dtype=float)
        return q / np.linalg.norm(q)

    @staticmethod
    def quat_to_matrix(q):
        x, y, z, w = q
        return np.array([
            [1 - 2*(y*y + z*z),   2*(x*y - z*w),     2*(x*z + y*w)],
            [2*(x*y + z*w),       1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w),       2*(y*z + x*w),     1 - 2*(x*x + y*y)]
        ])

    def compute_interval(self):
        times = self.df['time'].to_numpy()
        dt = np.min(np.diff(times))
        if dt <= 0:
            fps_real = self.max_fps
        else:
            fps_real = 1.0 / dt
        fps = max(min(fps_real, self.max_fps), self.min_fps)
        return 1000.0 / fps

    def animate(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("Rigid Body Path & Orientation")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        positions = np.vstack(self.df['position'])
        pad = 1.0
        ax.set_xlim(positions[:, 0].min() - pad, positions[:, 0].max() + pad)
        ax.set_ylim(positions[:, 1].min() - pad, positions[:, 1].max() + pad)
        ax.set_zlim(positions[:, 2].min() - pad, positions[:, 2].max() + pad)

        ax.plot(positions[:, 0], positions[:, 1],
                positions[:, 2], 'k--', alpha=0.3)

        quivers = {
            'x': ax.quiver(0, 0, 0, 0, 0, 0, color='r'),
            'y': ax.quiver(0, 0, 0, 0, 0, 0, color='g'),
            'z': ax.quiver(0, 0, 0, 0, 0, 0, color='b'),
        }

        def update(i):
            row = self.df.iloc[i]
            pos = np.array(row['position'])
            quat = self.normalize_quat(row['orientation'])
            Rm = self.quat_to_matrix(quat)
            axes = Rm @ np.eye(3) * self.axis_length

            for arr in quivers.values():
                arr.remove()

            quivers['x'] = ax.quiver(*pos, *axes[:, 0], color='r')
            quivers['y'] = ax.quiver(*pos, *axes[:, 1], color='g')
            quivers['z'] = ax.quiver(*pos, *axes[:, 2], color='b')
            return list(quivers.values())

        interval = self.compute_interval()

        ani = FuncAnimation(
            fig,
            update,
            frames=len(self.df),
            interval=interval,
            blit=False
        )
        plt.show()
