# -*- coding: utf-8 -*-
"""
Алгоритм комплексирования ИНС (инерциальная навигационная система) и ГНСС
в суммарную интегрированную навигационную систему на основе фильтра Калмана.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class FilterConfig:
    """Параметры фильтра и шумов."""
    dt: float                    # шаг дискретизации, с
    sigma_acc: float             # СКО шума ускорения ИНС, м/с^2
    sigma_gnss_pos: float        # СКО шума положения ГНСС, м
    sigma_gnss_vel: float        # СКО шума скорости ГНСС (если есть), м/с
    use_gnss_velocity: bool = True  # использовать ли скорость от ГНСС в измерении


# Состояние и измерения в ENU: индекс 0=East, 1=North, 2=Up (м, м/с).

class INSGNSSKalmanFilter:
    """
    Фильтр Калмана для комплексирования ИНС и ГНСС.
    Состояние x = [p_E, p_N, p_U, v_E, v_N, v_U] в ENU. Прогноз по ускорению ИНС в ENU, коррекция по ГНСС.
    """

    def __init__(self, config: FilterConfig, initial_state: np.ndarray):
        self.config = config
        self.dim_state = 6
        self.x = np.asarray(initial_state, dtype=float).ravel()
        if self.x.size != self.dim_state:
            raise ValueError(f"initial_state должен иметь размер {self.dim_state}, получен {self.x.size}")
        # Модель: p_{k+1} = p_k + v_k*dt + 0.5*a_k*dt^2, v_{k+1} = v_k + a_k*dt
        dt = config.dt
        self.F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ])

        # Матрица управления: вход — ускорение [ax, ay, az]
        self.B = np.array([
            [0.5 * dt**2, 0, 0],
            [0, 0.5 * dt**2, 0],
            [0, 0, 0.5 * dt**2],
            [dt, 0, 0],
            [0, dt, 0],
            [0, 0, dt],
        ])

        # Шум процесса (модель ошибок ИНС): Q
        sigma_a = config.sigma_acc
        q = sigma_a ** 2
        # Упрощённая Q для модели с ускорением как белый шум
        G = np.vstack([self.B[:3], self.B[3:] * 0.5])
        self.Q = (G * q) @ G.T
        self.Q += np.eye(6) * 1e-10  # для численной устойчивости

        # Ковариация ошибки состояния
        self.P = np.eye(6) * 10.0   # начальная неопределённость

        # Сохраняем конфиг для пересчёта F, B, Q при переменном dt
        self._sigma_acc = config.sigma_acc

        # Матрица наблюдения: ГНСС положение (и при необходимости скорость)
        if config.use_gnss_velocity:
            self.H = np.eye(6)  # измеряем и позицию, и скорость
            self.dim_meas = 6
        else:
            self.H = np.array([
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
            ])
            self.dim_meas = 3

        # Ковариация шума измерений ГНСС
        self.R = np.eye(self.dim_meas)
        self.R[:3, :3] *= config.sigma_gnss_pos ** 2
        if config.use_gnss_velocity:
            self.R[3:, 3:] *= config.sigma_gnss_vel ** 2

    def _set_dt(self, dt: float) -> None:
        """Пересчёт F, B, Q для заданного шага dt (для переменного шага по времени)."""
        self.F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ])
        self.B = np.array([
            [0.5 * dt**2, 0, 0],
            [0, 0.5 * dt**2, 0],
            [0, 0, 0.5 * dt**2],
            [dt, 0, 0],
            [0, dt, 0],
            [0, 0, dt],
        ])
        sigma_a = self._sigma_acc
        q = sigma_a ** 2
        G = np.vstack([self.B[:3], self.B[3:] * 0.5])
        self.Q = (G * q) @ G.T + np.eye(6) * 1e-10

    def predict(self, acceleration: np.ndarray, dt: Optional[float] = None) -> np.ndarray:
        """
        Шаг прогноза по ИНС: экстраполяция состояния по модели с входным ускорением.

        Parameters
        ----------
        acceleration : array (3,) — ускорение в НЭЗ [ax, ay, az], м/с^2
        dt : шаг по времени, с. Если задан — пересчитываются F, B, Q для этого шага
             (корректно при неравномерной сетке времени). Если None — используется config.dt.

        Returns
        -------
        x_pred : текущая оценка состояния после прогноза
        """
        step = self.config.dt if dt is None else float(np.clip(dt, 1e-6, 10.0))
        self._set_dt(step)
        acc = np.asarray(acceleration, dtype=float).ravel()
        acc = np.resize(acc, 3)  # [a_E, a_N, a_U] в ENU
        self.x = self.F @ self.x + self.B @ acc
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()

    def update(self, z_gnss: np.ndarray) -> np.ndarray:
        """
        Шаг коррекции по измерениям ГНСС (положение и при необходимости скорость).

        Parameters
        ----------
        z_gnss : array (3,) или (6,) — положение [px, py, pz] или [px, py, pz, vx, vy, vz]

        Returns
        -------
        x_corr : оценка состояния после коррекции
        """
        z = np.asarray(z_gnss, dtype=float).ravel()
        if z.size != self.dim_meas:
            z = np.resize(z, self.dim_meas)
        z_pred = self.H @ self.x
        y = z - z_pred
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.solve(S, np.eye(S.shape[0]))
        self.x = self.x + K @ y  # коррекция по измерению ГНСС (pos и при необходимости vel в ENU)
        I_KH = np.eye(6) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + (K @ self.R @ K.T)
        return self.x.copy()

    def get_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Возвращает текущую оценку состояния и ковариацию."""
        return self.x.copy(), self.P.copy()

    def get_position(self) -> np.ndarray:
        return self.x[:3].copy()

    def get_velocity(self) -> np.ndarray:
        return self.x[3:6].copy()


def run_ins_gnss_complex(
    time_grid: np.ndarray,
    acc_ins: np.ndarray,
    pos_gnss: np.ndarray,
    vel_gnss: Optional[np.ndarray] = None,
    config: Optional[FilterConfig] = None,
    x0: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Запуск комплексирования ИНС и ГНСС на всей траектории.

    Parameters
    ----------
    time_grid : (N,) массив моментов времени, с
    acc_ins : (N, 3) ускорения от ИНС в НЭЗ, м/с^2
    pos_gnss : (N, 3) положения от ГНСС, м (могут быть NaN при срыве)
    vel_gnss : (N, 3) или None — скорости от ГНСС
    config : параметры фильтра; по умолчанию dt из time_grid
    x0 : начальное состояние [px, py, pz, vx, vy, vz]

    Returns
    -------
    positions : (N, 3) оценённые положения суммарной системы
    velocities : (N, 3) оценённые скорости
    covariances : (N, 6, 6) ковариации ошибки состояния (опционально можно сократить)
    """
    n = len(time_grid)
    dt = float(np.median(np.diff(time_grid))) if n > 1 else 0.01
    if config is None:
        config = FilterConfig(
            dt=dt,
            sigma_acc=0.5,
            sigma_gnss_pos=2.0,
            sigma_gnss_vel=0.3,
            use_gnss_velocity=vel_gnss is not None,
        )
    if x0 is None:
        # Инициализация по первому доступному ГНСС
        valid = np.isfinite(pos_gnss).all(axis=1)
        if valid.any():
            i0 = np.where(valid)[0][0]
            p0 = pos_gnss[i0]
            v0 = vel_gnss[i0] if vel_gnss is not None else np.zeros(3)
        else:
            p0 = np.zeros(3)
            v0 = np.zeros(3)
        x0 = np.concatenate([p0, v0])

    kf = INSGNSSKalmanFilter(config, x0)
    positions = np.zeros((n, 3))   # (N, 3) ENU
    velocities = np.zeros((n, 3))
    covariances = np.zeros((n, 6, 6))

    for i in range(n):
        if i > 0:
            dt_i = time_grid[i] - time_grid[i - 1]
            dt_i = np.clip(dt_i, 1e-6, 10.0)
            kf.predict(acc_ins[i], dt=dt_i)  # прогноз с t_{i-1} на t_i по ускорению a_i
        positions[i] = kf.get_position()
        velocities[i] = kf.get_velocity()
        _, covariances[i] = kf.get_state()
        pos_ok = np.isfinite(pos_gnss[i]).all()
        if config.use_gnss_velocity and vel_gnss is not None:
            vel_ok = np.isfinite(vel_gnss[i]).all()
            if pos_ok and vel_ok:
                z = np.concatenate([pos_gnss[i], vel_gnss[i]])
                kf.update(z)
                positions[i] = kf.get_position()
                velocities[i] = kf.get_velocity()
                _, covariances[i] = kf.get_state()
        elif pos_ok:
            kf.update(pos_gnss[i])
            positions[i] = kf.get_position()
            velocities[i] = kf.get_velocity()
            _, covariances[i] = kf.get_state()
        # при невалидном ГНСС остаётся только прогноз

    return positions, velocities, covariances


def run_ins_gnss_hard_coupled(
    time_grid: np.ndarray,
    acc_ins: np.ndarray,
    pos_gnss: np.ndarray,
    vel_gnss: Optional[np.ndarray] = None,
    x0: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Жёсткая связь ИНС и ГНСС:
    - прогноз по ИНС на каждом шаге (интеграция ускорения),
    - при валидном ГНСС позиция ИНС принудительно ставится равной ГНСС,
    - при валидной скорости ГНСС (если передана) скорость также принудительно обновляется.

    Это не фильтр Калмана, а детерминированная схема «predict + hard reset by GNSS».
    """
    n = len(time_grid)
    acc = np.nan_to_num(np.asarray(acc_ins, dtype=float), nan=0.0)
    pos = np.asarray(pos_gnss, dtype=float)
    vel = None if vel_gnss is None else np.asarray(vel_gnss, dtype=float)

    positions = np.zeros((n, 3))
    velocities = np.zeros((n, 3))

    if x0 is not None:
        x0 = np.asarray(x0, dtype=float).ravel()
        p = np.resize(x0[:3], 3)
        v = np.resize(x0[3:6], 3)
    else:
        valid = np.isfinite(pos).all(axis=1)
        if valid.any():
            i0 = int(np.where(valid)[0][0])
            p = pos[i0].copy()
            if vel is not None and np.isfinite(vel[i0]).all():
                v = vel[i0].copy()
            else:
                v = np.zeros(3)
        else:
            p = np.zeros(3)
            v = np.zeros(3)

    positions[0] = p
    velocities[0] = v

    for i in range(1, n):
        dt_i = float(np.clip(time_grid[i] - time_grid[i - 1], 1e-6, 10.0))
        # Прогноз ИНС
        v = v + acc[i] * dt_i
        p = p + velocities[i - 1] * dt_i + 0.5 * acc[i] * dt_i * dt_i

        # Жёсткая коррекция по ГНСС
        if np.isfinite(pos[i]).all():
            p = pos[i].copy()
        if vel is not None and np.isfinite(vel[i]).all():
            v = vel[i].copy()

        positions[i] = p
        velocities[i] = v

    return positions, velocities
