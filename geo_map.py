# -*- coding: utf-8 -*-
"""
Привязка траекторий к карте: перевод локальных координат (м) в широту/долготу и отображение на карте.
Вход: опорные координаты (широта, долгота, высота); траектории в локальной системе East-North-Up (м).
Выход: интерактивная карта с положением траекторий.
"""

import numpy as np
from typing import Optional, List, Tuple

# Радиус Земли (WGS84), м
EARTH_RADIUS = 6_371_009.0


def enu_to_latlonalt(
    east: np.ndarray,
    north: np.ndarray,
    up: np.ndarray,
    lat0: float,
    lon0: float,
    alt0: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Перевод из локальной системы East-North-Up (метры) в широту, долготу, высоту (WGS84).

    Parameters
    ----------
    east, north, up : массивы или скаляры, м (X = East, Y = North, Z = Up)
    lat0, lon0 : широта и долгота опорной точки, градусы
    alt0 : высота опорной точки, м

    Returns
    -------
    lat, lon, alt : массивы (или скаляры) в градусах и метрах
    """
    east = np.atleast_1d(np.asarray(east, dtype=float))
    north = np.atleast_1d(np.asarray(north, dtype=float))
    up = np.atleast_1d(np.asarray(up, dtype=float))
    lat0_rad = np.deg2rad(lat0)
    lon0_rad = np.deg2rad(lon0)
    # Линейное приближение для малых расстояний
    dlat = north / EARTH_RADIUS
    dlon = east / (EARTH_RADIUS * np.cos(lat0_rad))
    lat_rad = lat0_rad + dlat
    lon_rad = lon0_rad + dlon
    lat = np.rad2deg(lat_rad)
    lon = np.rad2deg(lon_rad)
    alt = alt0 + up
    return lat, lon, alt


def pos_enu_to_latlonalt(
    pos_enu: np.ndarray,
    lat0: float,
    lon0: float,
    alt0: float = 0.0,
) -> np.ndarray:
    """
    Перевод массива положений (N, 3) из ENU [East, North, Up] в (N, 3) [lat, lon, alt].

    pos_enu : (N, 3) — восток, север, высота, м
    """
    pos_enu = np.asarray(pos_enu)
    if pos_enu.ndim == 1:
        pos_enu = pos_enu.reshape(1, -1)
    east, north, up = pos_enu[:, 0], pos_enu[:, 1], pos_enu[:, 2]
    lat, lon, alt = enu_to_latlonalt(east, north, up, lat0, lon0, alt0)
    return np.column_stack([lat, lon, alt])


def latlonalt_to_enu(
    lat: np.ndarray,
    lon: np.ndarray,
    alt: np.ndarray,
    lat0: float,
    lon0: float,
    alt0: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Перевод из широты, долготы, высоты (WGS84) в локальную ENU относительно (lat0, lon0, alt0).
    """
    lat = np.atleast_1d(np.asarray(lat, dtype=float))
    lon = np.atleast_1d(np.asarray(lon, dtype=float))
    alt = np.atleast_1d(np.asarray(alt, dtype=float))
    lat0_rad = np.deg2rad(lat0)
    north = np.deg2rad(lat - lat0) * EARTH_RADIUS
    east = np.deg2rad(lon - lon0) * EARTH_RADIUS * np.cos(lat0_rad)
    up = alt - alt0
    return east, north, up


def latlonalt_to_pos_enu(
    latlonalt: np.ndarray,
    lat0: float,
    lon0: float,
    alt0: float = 0.0,
) -> np.ndarray:
    """(N, 3) [lat, lon, alt] -> (N, 3) [east, north, up] в метрах."""
    latlonalt = np.asarray(latlonalt)
    if latlonalt.ndim == 1:
        latlonalt = latlonalt.reshape(1, -1)
    e, n, u = latlonalt_to_enu(
        latlonalt[:, 0], latlonalt[:, 1], latlonalt[:, 2], lat0, lon0, alt0
    )
    return np.column_stack([e, n, u])


def build_map_html(
    latlon_true: Optional[np.ndarray] = None,
    latlon_ins: Optional[np.ndarray] = None,
    latlon_combined: Optional[np.ndarray] = None,  # обязателен при вызове: суммарная траектория (красная линия)
    latlon_gnss: Optional[np.ndarray] = None,
    center_lat: Optional[float] = None,
    center_lon: Optional[float] = None,
    output_path: str = "ins_gnss_map.html",
    title: str = "Траектория: ИНС и ГНСС на карте",
) -> str:
    """
    Строит интерактивную карту (OpenStreetMap) с траекториями и сохраняет в HTML.

    Parameters
    ----------
    latlon_true : (N, 2) или (N, 3), опционально — истинная траектория (для симуляции). Если None или пустой — синяя линия не рисуется.
    latlon_ins : (N, 2) или (N, 3), опционально — только ИНС
    latlon_combined : (N, 2) или (N, 3) — суммарная система (ИНС+ГНСС), красная линия на карте
    latlon_gnss : (N, 2) или (N, 3), опционально — измерения ГНСС (могут содержать NaN)
    center_lat, center_lon : центр карты; если None — по центру траектории
    output_path : путь к сохраняемому файлу
    title : заголовок страницы

    Returns
    -------
    output_path : путь к сохранённому файлу
    """
    try:
        import folium
    except ImportError:
        raise ImportError("Установите folium: pip install folium") from None

    def to_list(arr: np.ndarray) -> List[Tuple[float, float]]:
        arr = np.asarray(arr)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        # Убираем строки с NaN
        valid = np.isfinite(arr).all(axis=1)
        pts = arr[valid, :2].tolist()
        return [(float(p[0]), float(p[1])) for p in pts]

    points_true = to_list(latlon_true) if latlon_true is not None and np.asarray(latlon_true).size > 0 else []
    points_ins = to_list(latlon_ins) if latlon_ins is not None and np.asarray(latlon_ins).size > 0 else []
    points_combined = to_list(latlon_combined) if latlon_combined is not None else []
    points_gnss = to_list(latlon_gnss) if latlon_gnss is not None else []
    all_pts = points_true + points_ins + points_combined + points_gnss
    all_lat = [p[0] for p in all_pts]
    all_lon = [p[1] for p in all_pts]
    if not all_lat:
        all_lat, all_lon = [55.75], [37.62]  # Москва по умолчанию
    if center_lat is None:
        center_lat = (min(all_lat) + max(all_lat)) / 2
    if center_lon is None:
        center_lon = (min(all_lon) + max(all_lon)) / 2

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=15,
        tiles="OpenStreetMap",
        control_scale=True,
    )
    folium.TileLayer("CartoDB positron", name="Светлая").add_to(m)
    folium.TileLayer("CartoDB dark_matter", name="Тёмная").add_to(m)
    folium.LayerControl().add_to(m)

    if len(points_true) >= 2:
        folium.PolyLine(
            points_true,
            color="blue",
            weight=4,
            opacity=0.9,
            popup="Истинная траектория",
        ).add_to(m)
    if len(points_ins) >= 2:
        folium.PolyLine(
            points_ins,
            color="purple",
            weight=3,
            opacity=0.8,
            dash_array="5, 5",
            popup="Только ИНС",
        ).add_to(m)
    if len(points_combined) >= 2:
        folium.PolyLine(
            points_combined,
            color="red",
            weight=4,
            opacity=0.9,
            popup="Суммарная (ИНС+ГНСС)",
        ).add_to(m)
    if len(points_gnss) >= 2:
        folium.PolyLine(
            points_gnss,
            color="green",
            weight=2,
            opacity=0.6,
            popup="ГНСС",
        ).add_to(m)

    start_end_pts = points_true or points_combined
    if start_end_pts:
        folium.Marker(
            start_end_pts[0],
            popup="Старт",
            icon=folium.Icon(color="green", icon="play"),
        ).add_to(m)
        folium.Marker(
            start_end_pts[-1],
            popup="Финиш",
            icon=folium.Icon(color="red", icon="stop"),
        ).add_to(m)

    m.get_root().html.add_child(folium.Element(f"<title>{title}</title>"))
    m.save(output_path)
    return output_path
