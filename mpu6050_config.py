# -*- coding: utf-8 -*-
"""
MPU6050 configuration for INS/GNSS fusion (EKF). Output 10 Hz.

1) PARAMETER EFFECTS
--------------------
- Dynamic range: max measurable |a| or |ω| before saturation. Higher range => less
  resolution (fewer LSB per physical unit) and usually similar or slightly worse
  noise density (noise is roughly range-independent in LSB for many MEMS).

- Sensitivity (LSB scaling): LSB/g or LSB/(deg/s). ±2g => 16384 LSB/g; ±4g => 8192;
  ±8g => 4096; ±16g => 2048. Gyro ±250 => 131 LSB/(deg/s); ±500 => 65.5; ±1000 => 32.8;
  ±2000 => 16.4. Resolution = range / (2^16) per LSB; quantization noise ~ 1/sqrt(12) LSB.

- Noise density: typ. accel ~300–400 µg/√Hz (slightly higher at ±16g); gyro
  ~0.005–0.01 °/s/√Hz. After DLPF, RMS noise = density * sqrt(BW). Lower DLPF BW
  => less noise but more phase lag and more in-band attenuation of real dynamics.

- DLPF latency: group delay ~ 1/(2*pi*BW) to ~ 1/(pi*BW) depending on order.
  21 Hz => ~7.6–15 ms. 94 Hz => ~1.7–3.4 ms. Affects temporal alignment with GNSS
  and double-integration (latency in accel => position lag).

- Aliasing: Internal sample rate 1 kHz (or 8 kHz). At 10 Hz output, downsampling
  by 100x. Without DLPF, any signal above 5 Hz folds into 0–5 Hz. DLPF < 5 Hz
  (e.g. 5 Hz or 10 Hz) limits foldover; 21 Hz still allows some aliasing above
  Nyquist (5 Hz) but attenuates it. Rule: DLPF BW << output_rate/2 to reduce
  aliasing; trade-off is latency and phase lag.

2) PLATFORM RECOMMENDATIONS
---------------------------
- Ground vehicle: |a| rarely >1–2g (bumps, braking); |ω| often <100–200 deg/s.
  Accel ±4g (or ±8g if harsh); gyro ±500 deg/s; DLPF 21–44 Hz (smooth motion,
  latency acceptable). 21 Hz: lower noise, ~8 ms lag; 44 Hz: less lag, slightly
  noisier.

- Multirotor UAV: Short sharp accelerations (2–4g); high angular rates (200–500+ deg/s).
  Accel ±8g to avoid saturation on punch-out; gyro ±500 or ±1000 deg/s; DLPF 44–94 Hz
  to track fast dynamics. 21 Hz too slow for aggressive manoeuvres (phase lag
  degrades attitude and a_nav).

- Lab (low dynamics): |a| ≈ 1g + small perturbations; |ω| small. Accel ±2g (best
  resolution); gyro ±250 deg/s; DLPF 10–21 Hz for minimal noise, bias estimation
  in EKF benefits from low noise.

3) TRADE-OFFS
-------------
- Saturation vs resolution: Larger range avoids clipping but halves resolution per
  bit and can worsen effective SNR for small signals. For EKF, saturation causes
  large innovations and can destabilise; prefer headroom over resolution.

- Noise vs bandwidth: Lower DLPF => lower RMS noise => better velocity/position
  from integration and better bias observability in EKF. Higher DLPF => less
  phase lag and better tracking of fast motion. EKF stability: very low BW
  (e.g. 5 Hz) can make dynamics look sluggish and increase model mismatch;
  very high BW (260 Hz) increases noise and can make Q tuning harder.

- Bias estimation: Bias observability in EKF improves with low noise and
  persistent excitation (motion). Low DLPF reduces noise and helps; too low BW
  can filter out useful motion and slow bias convergence.

4) IMPROVED CONFIGURATION (generic default)
-------------------------------------------
Current: ±4g, ±500 deg/s, 21 Hz.
- Accel ±4g: reasonable for ground and moderate UAV; use ±8g only if expecting
  sustained >2g (e.g. aggressive UAV). ±4g gives 8192 LSB/g => 0.5 mg/LSB.
- Gyro ±500 deg/s: good for ground and most UAV; ±250 if only lab/low dynamics
  (better resolution and often lower noise).
- DLPF: 21 Hz is conservative (low noise, ~8 ms lag). For 10 Hz fusion, signal
  BW of interest is <5 Hz; 21 Hz is above Nyquist so some aliasing possible.
  Safer: 10 Hz DLPF (strong anti-aliasing, more latency ~16 ms) or keep 21 Hz
  as compromise. For UAV or high dynamics, 44 Hz reduces phase lag.
"""

from dataclasses import dataclass
from typing import Tuple

# MPU6050 LSB scaling (datasheet)
# Accel: LSB per g (sensitivity). ±2g=16384, ±4g=8192, ±8g=4096, ±16g=2048
ACCEL_LSB_PER_G = {
    "2g": 16384.0,
    "4g": 8192.0,
    "8g": 4096.0,
    "16g": 2048.0,
}
# Gyro: LSB per deg/s. ±250=131, ±500=65.5, ±1000=32.8, ±2000=16.4
GYRO_LSB_PER_DEG_S = {
    "250": 131.0,
    "500": 65.5,
    "1000": 32.8,
    "2000": 16.4,
}

# Typical noise density (order of magnitude). Accel µg/√Hz, Gyro °/s/√Hz
# Used only for rough scaling; actual part-to-part variation is large.
ACCEL_NOISE_DENSITY_UG_SQRT_HZ = 400.0
GYRO_NOISE_DENSITY_DEG_S_SQRT_HZ = 0.008


@dataclass
class MPU6050Scales:
    """Scale factors (physical per LSB) for a given range. g in m/s²: 9.80665."""
    accel_lsb_per_g: float   # LSB per g
    gyro_lsb_per_deg_s: float  # LSB per deg/s
    G: float = 9.80665

    @property
    def accel_scale(self) -> float:
        """(m/s²) per LSB."""
        return self.G / self.accel_lsb_per_g

    @property
    def gyro_scale_rad_s(self) -> float:
        """(rad/s) per LSB."""
        return (3.141592653589793 / 180.0) / self.gyro_lsb_per_deg_s


def get_scales_for_ranges(accel_range_g: str = "4g", gyro_range_deg_s: str = "500") -> MPU6050Scales:
    """accel_range_g: '2g'|'4g'|'8g'|'16g'. gyro_range_deg_s: '250'|'500'|'1000'|'2000'."""
    return MPU6050Scales(
        accel_lsb_per_g=ACCEL_LSB_PER_G[accel_range_g],
        gyro_lsb_per_deg_s=GYRO_LSB_PER_DEG_S[gyro_range_deg_s],
    )


# Recommended presets for 10 Hz INS/GNSS fusion
PRESET_GROUND = ("4g", "500")   # ±4g, ±500 deg/s, DLPF 21–44 Hz in firmware
PRESET_UAV = ("8g", "500")      # ±8g, ±500 deg/s, DLPF 44–94 Hz
PRESET_LAB = ("2g", "250")      # ±2g, ±250 deg/s, DLPF 10–21 Hz

# If firmware uses ±500 deg/s: gyro_bias_LSB = bias_deg_s * 65.5 (not 131).
# imu_body_to_nav.ArduinoIMUScales uses 131 (±250); for ±500 set gyro_bias
# and gyro_scale from get_scales_for_ranges("4g","500") or match firmware range.
