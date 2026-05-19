# -*- coding: utf-8 -*-
"""
Симуляция траектории и измерений ИНС/ГНСС для проверки алгоритма комплексирования.

Алгоритмы:
- Генерация истинной траектории (окружность в XY + линейный подъём по Z).
- Модель ошибок ИНС: постоянный bias + белый шум по ускорению.
- Модель ГНСС: белый шум по положению/скорости, имитация срыва фиксации на интервале.
- Интеграция «только ИНС»: метод Эйлера (прямоугольники) для скорости и позиции.
- Режим body frame: обратный проход от истинного ускорения в нав. СК к «сырым» body,
  затем BodyToNavProcessor (Пуассон + Родрига) даёт ускорение в нав. СК для фильтра.
- Комплексирование: Калман (прогноз по ИНС, коррекция по ГНСС), вывод карты и графиков.
"""

import numpy as np
from ins_gnss_filter import FilterConfig, run_ins_gnss_complex
from geo_map import pos_enu_to_latlonalt, build_map_html
from imu_body_to_nav import BodyToNavProcessor, rotation_body_to_nav, GRAVITY


def generate_trajectory(
    duration: float = 60.0,
    dt: float = 0.1,
    seed: int = 42,
) -> tuple:
    """
    Генерация истинной траектории и имитация измерений ИНС и ГНСС.

    Алгоритм траектории: движение по окружности в плоскости XY (радиус R, угловая скорость ω),
    плюс равномерный подъём по Z. Истинные pos, vel, acc вычисляются аналитически.
    ИНС: к true_acc добавляется постоянный bias и белый шум (имитация ошибок датчика).
    ГНСС: к true_pos и true_vel — белый шум; на интервале [loss_start, loss_end) — NaN (срыв фиксации).

    Returns
    -------
    t : (N,) время, с
    true_pos : (N, 3) истинное положение, м
    true_vel : (N, 3) истинная скорость, м/с
    true_acc : (N, 3) истинное ускорение, м/с^2
    ins_acc : (N, 3) ускорение от ИНС (с ошибками), м/с^2
    gnss_pos : (N, 3) положение от ГНСС (с шумом, возможны пропуски), м
    gnss_vel : (N, 3) скорость от ГНСС (с шумом), м/с
    """
    rng = np.random.default_rng(seed)
    n = int(round(duration / dt)) + 1
    t = np.linspace(0, duration, n)

    # --- Истинная траектория: окружность в XY + линейный подъём по Z ---
    # Кинематика: r(t) = (R*cos(ωt), R*sin(ωt), v_z*t), v = dr/dt, a = d²r/dt²
    omega = 2 * np.pi / 30  # период ~30 с, рад/с
    R = 50.0  # радиус окружности, м
    v_xy = R * omega
    true_pos = np.zeros((n, 3))
    true_vel = np.zeros((n, 3))
    true_acc = np.zeros((n, 3))

    for i in range(n):
        theta = omega * t[i]
        true_pos[i, 0] = R * np.cos(theta)
        true_pos[i, 1] = R * np.sin(theta)
        true_pos[i, 2] = 2 * t[i]  # медленный подъём, м
        true_vel[i, 0] = -R * omega * np.sin(theta)
        true_vel[i, 1] = R * omega * np.cos(theta)
        true_vel[i, 2] = 2.0
        true_acc[i, 0] = -R * omega**2 * np.cos(theta)  # центростремительное + 0 по Z
        true_acc[i, 1] = -R * omega**2 * np.sin(theta)
        true_acc[i, 2] = 0.0

    # --- Модель ошибок ИНС: постоянный bias (калибровка) + белый шум по ускорению ---
    acc_bias = 0.05 * (rng.standard_normal(3))
    ins_acc = true_acc + acc_bias + 0.1 * rng.standard_normal((n, 3))

    # --- Модель измерений ГНСС: аддитивный белый шум + имитация срыва фиксации ---
    sigma_pos_gnss = 1.5   # СКО положения, м
    sigma_vel_gnss = 0.2   # СКО скорости, м/с
    gnss_pos = true_pos + sigma_pos_gnss * rng.standard_normal((n, 3))
    gnss_vel = true_vel + sigma_vel_gnss * rng.standard_normal((n, 3))
    loss_start = int(0.3 * n)
    loss_end = int(0.5 * n)
    gnss_pos[loss_start:loss_end, :] = np.nan
    gnss_vel[loss_start:loss_end, :] = np.nan

    return t, true_pos, true_vel, true_acc, ins_acc, gnss_pos, gnss_vel


def integrate_ins_only(
    time_grid: np.ndarray,
    ins_acc: np.ndarray,
    pos0: np.ndarray,
    vel0: np.ndarray,
) -> tuple:
    """
    Траектория только по ИНС: интегрирование ускорений без коррекции ГНСС.

    Алгоритм: дискретная модель «прямоугольники» (как в фильтре Калмана).
    На шаге i: v_i = v_{i-1} + a_{i-1}*dt, p_i = p_{i-1} + v_{i-1}*dt + 0.5*a_{i-1}*dt².
    Ускорение a_{i-1} относится к интервалу [t_{i-1}, t_i]. Начальные p0, v0 задаются снаружи.

    Returns
    -------
    positions : (N, 3) положение по ИНС, м
    velocities : (N, 3) скорость по ИНС, м/с
    """
    n = len(time_grid)
    positions = np.zeros((n, 3))
    velocities = np.zeros((n, 3))
    positions[0] = np.asarray(pos0, dtype=float).ravel()[:3]
    velocities[0] = np.asarray(vel0, dtype=float).ravel()[:3]
    for i in range(1, n):
        dt = time_grid[i] - time_grid[i - 1]
        velocities[i] = velocities[i - 1] + ins_acc[i - 1] * dt
        positions[i] = positions[i - 1] + velocities[i - 1] * dt + 0.5 * ins_acc[i - 1] * dt**2
    return positions, velocities


def body_frame_to_nav_accel_sequence(
    time_grid: np.ndarray,
    true_acc_nav: np.ndarray,
    true_roll: np.ndarray,
    true_pitch: np.ndarray,
    acc_bias: np.ndarray,
    acc_noise_std: float,
    gyro_noise_std: float,
    seed: int = 42,
) -> tuple:
    """
    Симуляция в связанных осях (body frame): курс в начале = 0, ориентация интегрируется по гироскопу.

    Алгоритм (обратный проход для получения «сырых» ИМУ):
    1) По истинному ускорению в нав. СК и углам roll, pitch строим удельную силу в нав. СК:
       f_nav = true_acc_nav + g_nav (g_nav = (0,0,-g)).
    2) Переводим в body: f_body = R_nb.T @ f_nav (R_nb из rotation_body_to_nav).
    3) Имитируем ошибки датчика: f_body += acc_bias + белый шум.
    4) Гироскоп: угловые скорости из конечных разностей roll, pitch плюс белый шум; курс = 0.
    5) BodyToNavProcessor обновляет ориентацию по Пуассону (Родрига) и переводит f_body → acc_nav.
    Возвращает (acc_nav, gyro_body) для подачи в фильтр и в лог.

    true_acc_nav : (N, 3) м/с² в НЭЗ (ENU)
    true_roll, true_pitch : (N,) рад (например нули)
    Возвращает (acc_nav, gyro_body): (N, 3) ускорение в НЭЗ, (N, 3) гироскоп в body, рад/с.
    """
    rng = np.random.default_rng(seed)
    n = len(time_grid)
    dt = float(np.median(np.diff(time_grid))) if n > 1 else 0.1
    g_nav = np.array([0.0, 0.0, -GRAVITY])
    processor = BodyToNavProcessor(initial_yaw=0.0)
    acc_nav_out = np.zeros((n, 3))
    gyro_out = np.zeros((n, 3))
    for i in range(n):
        roll, pitch = float(true_roll[i]), float(true_pitch[i])
        R_nb = rotation_body_to_nav(roll, pitch, 0.0)
        f_nav = true_acc_nav[i] + g_nav
        f_body = (R_nb.T) @ f_nav
        f_body += acc_bias + acc_noise_std * rng.standard_normal(3)
        if i + 1 < n:
            roll_dot = (true_roll[i + 1] - true_roll[i]) / (time_grid[i + 1] - time_grid[i])
            pitch_dot = (true_pitch[i + 1] - true_pitch[i]) / (time_grid[i + 1] - time_grid[i])
        else:
            roll_dot = pitch_dot = 0.0
        wx = roll_dot + gyro_noise_std * rng.standard_normal()
        wy = pitch_dot + gyro_noise_std * rng.standard_normal()
        wz = 0.0 + gyro_noise_std * rng.standard_normal()
        gyro_out[i] = (wx, wy, wz)
        processor.update_orientation(wx, wy, wz, dt)
        acc_nav_out[i] = processor.accel_body_to_nav(f_body)
    return acc_nav_out, gyro_out


def main(
    lat0: float = 55.7558,
    lon0: float = 37.6173,
    alt0: float = 150.0,
):
    """
    Точка входа: генерирует траекторию, считает «только ИНС» и комплексирование ИНС+ГНСС,
    выводит ошибки, карту и графики.

    lat0, lon0, alt0 — опорная точка в WGS84 (привязка локальной ENU к карте).
    """
    dt = 0.1
    duration = 60.0
    use_body_frame = True  # True: симуляция body frame (обратный проход + Пуассон)
    t, true_pos, true_vel, true_acc, ins_acc_direct, gnss_pos, gnss_vel = generate_trajectory(
        duration=duration, dt=dt
    )
    if use_body_frame:
        true_roll = np.zeros(len(t))
        true_pitch = np.zeros(len(t))
        acc_bias = 0.05 * np.array([0.1, 0.1, 0.1])
        ins_acc, gyro_body = body_frame_to_nav_accel_sequence(
            t, true_acc, true_roll, true_pitch,
            acc_bias=acc_bias,
            acc_noise_std=0.1,
            gyro_noise_std=0.01,
        )
    else:
        ins_acc = ins_acc_direct
        gyro_body = np.zeros((len(t), 3))

    # Параметры фильтра Калмана: СКО шума ускорения ИНС и ГНСС (положение/скорость)
    config = FilterConfig(
        dt=dt,
        sigma_acc=0.15,
        sigma_gnss_pos=1.5,
        sigma_gnss_vel=0.2,
        use_gnss_velocity=True,
    )
    x0 = np.concatenate([gnss_pos[0], gnss_vel[0]])
    if not np.isfinite(x0).all():
        x0 = np.concatenate([true_pos[0], true_vel[0]])

    # Траектория «только ИНС» — та же дискретная модель, что в КФ, без коррекции по ГНСС
    ins_only_pos, ins_only_vel = integrate_ins_only(t, ins_acc, true_pos[0], true_vel[0])

    # Комплексирование: прогноз по ИНС (F, B), коррекция по ГНСС (H, R), переменный шаг
    positions, velocities, covariances = run_ins_gnss_complex(
        time_grid=t,
        acc_ins=ins_acc,
        pos_gnss=gnss_pos,
        vel_gnss=gnss_vel,
        config=config,
        x0=x0,
    )

    # Метрики: норма ошибки положения и скорости (суммарная система и только ИНС)
    err_pos = np.linalg.norm(positions - true_pos, axis=1)
    err_vel = np.linalg.norm(velocities - true_vel, axis=1)
    err_pos_ins = np.linalg.norm(ins_only_pos - true_pos, axis=1)
    err_vel_ins = np.linalg.norm(ins_only_vel - true_vel, axis=1)
    print("Комплексирование ИНС и ГНСС — результаты симуляции")
    print("--- Суммарная система (ИНС+ГНСС) ---")
    print("  Средняя ошибка положения (м):", np.nanmean(err_pos))
    print("  Средняя ошибка скорости (м/с):", np.nanmean(err_vel))
    print("  Макс. ошибка положения (м):", np.nanmax(err_pos))
    print("--- Только ИНС ---")
    print("  Средняя ошибка положения (м):", np.nanmean(err_pos_ins))
    print("  Средняя ошибка скорости (м/с):", np.nanmean(err_vel_ins))
    print("  Макс. ошибка положения (м):", np.nanmax(err_pos_ins))

    # Вывод значений гироскопа (и при необходимости сохранение в файл)
    if use_body_frame and gyro_body.size > 0:
        np.savetxt(
            "ins_gnss_imu_log.csv",
            np.hstack([t.reshape(-1, 1), positions, velocities, gyro_body]),
            delimiter=",",
            header="t,px,py,pz,vx,vy,vz,gyro_x,gyro_y,gyro_z",
            comments="",
        )
        print("Лог ИМУ (позиция, скорость, гироскоп рад/с): ins_gnss_imu_log.csv")
        print("  Гироскоп (среднее по траектории), рад/с: wx={:.4f}, wy={:.4f}, wz={:.4f}".format(
            np.nanmean(gyro_body[:, 0]), np.nanmean(gyro_body[:, 1]), np.nanmean(gyro_body[:, 2])
        ))

    # Преобразование ENU → широта/долгота/высота для отображения на карте (опора lat0, lon0, alt0)
    true_ll = pos_enu_to_latlonalt(true_pos, lat0, lon0, alt0)
    ins_only_ll = pos_enu_to_latlonalt(ins_only_pos, lat0, lon0, alt0)
    combined_ll = pos_enu_to_latlonalt(positions, lat0, lon0, alt0)
    gnss_ll = pos_enu_to_latlonalt(gnss_pos, lat0, lon0, alt0)

    try:
        build_map_html(
            latlon_true=true_ll,
            latlon_ins=ins_only_ll,
            latlon_combined=combined_ll,
            latlon_gnss=gnss_ll,
            output_path="ins_gnss_map.html",
            title="ИНС+ГНСС: траектория на карте",
        )
        print("Карта сохранена: ins_gnss_map.html")
    except Exception as e:
        print("Карта не построена (установите folium: pip install folium):", e)

    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))

        ax = axes[0, 0]
        ax.plot(true_pos[:, 0], true_pos[:, 1], "b-", label="Истина")
        ax.plot(gnss_pos[:, 0], gnss_pos[:, 1], "g.", markersize=2, alpha=0.7, label="ГНСС")
        ax.plot(ins_only_pos[:, 0], ins_only_pos[:, 1], "m--", label="Только ИНС")
        ax.plot(positions[:, 0], positions[:, 1], "r-", label="Суммарная (ИНС+ГНСС)")
        ax.set_xlabel("X, м")
        ax.set_ylabel("Y, м")
        ax.legend()
        ax.set_title("Траектория (вид сверху)")
        ax.axis("equal")
        ax.grid(True)

        ax = axes[0, 1]
        ax.plot(t, true_pos[:, 2], "b-", label="Истина")
        ax.plot(t, ins_only_pos[:, 2], "m--", label="Только ИНС")
        ax.plot(t, positions[:, 2], "r-", label="Суммарная")
        ax.set_xlabel("t, с")
        ax.set_ylabel("Высота Z, м")
        ax.legend()
        ax.set_title("Высота")
        ax.grid(True)

        ax = axes[1, 0]
        ax.plot(t, err_pos_ins, "m--", label="Только ИНС")
        ax.plot(t, err_pos, "r-", label="Суммарная (ИНС+ГНСС)")
        ax.set_xlabel("t, с")
        ax.set_ylabel("Ошибка положения, м")
        ax.set_title("Норма ошибки положения")
        ax.legend()
        ax.grid(True)

        ax = axes[1, 1]
        ax.plot(t, err_vel_ins, "m--", label="Только ИНС")
        ax.plot(t, err_vel, "r-", label="Суммарная (ИНС+ГНСС)")
        ax.set_xlabel("t, с")
        ax.set_ylabel("Ошибка скорости, м/с")
        ax.set_title("Норма ошибки скорости")
        ax.legend()
        ax.grid(True)

        plt.tight_layout()
        plt.savefig("ins_gnss_result.png", dpi=150)
        plt.show()
    except Exception as e:
        print("Визуализация не построена:", e)

    return t, true_pos, ins_only_pos, ins_only_vel, positions, velocities, covariances


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="ИНС+ГНСС: симуляция и карта. Вход — опорные координаты, выход — положение на карте.")
    p.add_argument("--lat", type=float, default=55.7558, help="Широта опорной точки, градусы")
    p.add_argument("--lon", type=float, default=37.6173, help="Долгота опорной точки, градусы")
    p.add_argument("--alt", type=float, default=150.0, help="Высота опорной точки, м")
    args = p.parse_args()
    main(lat0=args.lat, lon0=args.lon, alt0=args.alt)
