/*
 * 1) Интерактивная калибровка акселерометра MPU6050 по 6 положениям (реальные усреднённые raw-показания),
 *    затем калибровка смещения гироскопа в покое.
 * 2) Оценка дисперсии шума в статике ~10 минут в «выставленной» СК (после Ry/Rx по roll/pitch).
 *    Полученные var(ah_x), var(ah_y) в (м/с²)² можно использовать как элементы процессного шума Q
 *    для блоков, связанных со случайным ускорением (см. вывод в Serial и свой выбор дискретизации dt).
 *
 * Константы ACC_* из других файлов НЕ используются: bias и масштаб по осям ДАТЧИКА считаются здесь:
 *   bias_x = (max_ax + min_ax) / 2,  scale_x = (max_ax - min_ax) / (2*g)
 * (max/min — средние показания при +ось и −ось к небу), затем ax_cal = (raw - bias) / scale.
 *
 * Длительность усреднения в каждой позе задаётся POSE_COLLECT_MS (при желании поставьте 600000 для ~10 мин на позу).
 *
 * Serial 115200: после текста позы установите плату и отправьте любой символ (Enter) — начнётся сбор.
 */

#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

Adafruit_MPU6050 mpu;

// --- Физическая константа для масштаба (м/с²) ---
static const float G_MS2 = 9.80665f;

// Усреднение одной позы (мс). Для «долго держать» увеличьте, напр. 600000UL (~10 мин).
static const unsigned long POSE_COLLECT_MS = 15000UL;

static const unsigned long CALIBRATION_DURATION_MS = 5000UL;
static const unsigned long SETTLE_DURATION_MS = 3000UL;
static const unsigned long COLLECT_DURATION_MS = 10UL * 60UL * 1000UL;

static const float COMPL_ALPHA = 0.98f;

// Калибровка акселя по результатам 6-поз (ось датчика Adafruit / MPU6050 sensor frame)
float acc_bias_x = 0.0f, acc_bias_y = 0.0f, acc_bias_z = 0.0f;
float acc_scale_x = 1.0f, acc_scale_y = 1.0f, acc_scale_z = 1.0f;

// Гиро: смещение в °/с по трём каналам датчика (как в arduino_ins_gnss — сырое усреднение)
float cal_bx = 0.0f, cal_by = 0.0f, cal_bz = 0.0f;

enum Phase : uint8_t {
  PHASE_CAL_ACCEL,
  PHASE_CAL_GYRO,
  PHASE_SETTLE,
  PHASE_COLLECT,
  PHASE_DONE
};
Phase phase = PHASE_CAL_ACCEL;

enum AccelPoseSub : uint8_t { ACCEL_WAIT_ENTER, ACCEL_COLLECTING };

uint8_t accel_pose_idx = 0;
AccelPoseSub accel_sub = ACCEL_WAIT_ENTER;
unsigned long accel_pose_deadline_ms = 0;
float accel_sum_ax = 0.0f, accel_sum_ay = 0.0f, accel_sum_az = 0.0f;
uint32_t accel_pose_samples = 0;

struct PoseMean {
  float x, y, z;
};
PoseMean pose_mean[6];

unsigned long phaseStartMs = 0;
unsigned long lastAngleUs = 0;

float roll_deg = 0.0f, pitch_deg = 0.0f, yaw_deg = 0.0f;
float roll_prev = 0.0f, pitch_prev = 0.0f;

float sum_gx_raw = 0.0f, sum_gy_raw = 0.0f, sum_gz_raw = 0.0f;
int cal_count = 0;

float mean_ax_h = 0.0f, mean_ay_h = 0.0f;
float m2_ax_h = 0.0f, m2_ay_h = 0.0f;
uint32_t n_stats = 0;

unsigned long lastProgressMs = 0;

bool imuOk = false;

static const char *POSE_HINT(uint8_t i) {
  switch (i) {
    case 0:
      return "Поза 1/6: платформа ГОРИЗОНТАЛЬНО, чип MPU6050 СВЕРХУ — ось +Z направлена ВВЕРХ к небу.";
    case 1:
      return "Поза 2/6: переверните плату ВВЕРХ ДНОМ — ось +Z направлена ВНИЗ к столу.";
    case 2:
      return "Поза 3/6: поставьте РЕБРОМ так, чтобы ось +X была ВЕРТИКАЛЬНО ВВЕРХ.";
    case 3:
      return "Поза 4/6: ось +X ВЕРТИКАЛЬНО ВНИЗ.";
    case 4:
      return "Поза 5/6: ось +Y ВЕРТИКАЛЬНО ВВЕРХ.";
    case 5:
      return "Поза 6/6: ось +Y ВЕРТИКАЛЬНО ВНИЗ.";
    default:
      return "";
  }
}

static void flush_serial_input() {
  while (Serial.available() > 0)
    Serial.read();
}

static void body_from_sensor_cal(float ax_raw, float ay_raw, float az_raw,
                                 float &ax_b, float &ay_b, float &az_b) {
  float ax_cal = (ax_raw - acc_bias_x) / acc_scale_x;
  float ay_cal = (ay_raw - acc_bias_y) / acc_scale_y;
  float az_cal = (az_raw - acc_bias_z) / acc_scale_z;
  ax_b = az_cal;
  ay_b = -ay_cal;
  az_b = ax_cal;
}

static void accel_body_to_level(float ax_b, float ay_b, float az_b,
                                float roll_deg_in, float pitch_deg_in,
                                float &ah_x, float &ah_y, float &ah_z) {
  float phi = roll_deg_in * (float)PI / 180.0f;
  float theta = pitch_deg_in * (float)PI / 180.0f;
  float cr = cosf(phi), sr = sinf(phi);
  float ct = cosf(theta), st = sinf(theta);
  float x1 = ax_b;
  float y1 = cr * ay_b + sr * az_b;
  float z1 = -sr * ay_b + cr * az_b;
  ah_x = ct * x1 - st * z1;
  ah_y = y1;
  ah_z = st * x1 + ct * z1;
}

static void welford_update_pair(float x, float y, float &mean_x, float &mean_y,
                                float &m2_x, float &m2_y, uint32_t &n) {
  n++;
  float nf = (float)n;
  float dx = x - mean_x;
  mean_x += dx / nf;
  float dx2 = x - mean_x;
  m2_x += dx * dx2;
  float dy = y - mean_y;
  mean_y += dy / nf;
  float dy2 = y - mean_y;
  m2_y += dy * dy2;
}

static void print_banner(const char *msg) {
  Serial.println();
  Serial.println(msg);
}

static bool compute_accel_calibration_from_poses() {
  float max_ax = pose_mean[2].x;
  float min_ax = pose_mean[3].x;
  float max_ay = pose_mean[4].y;
  float min_ay = pose_mean[5].y;
  float max_az = pose_mean[0].z;
  float min_az = pose_mean[1].z;

  float span_ax = max_ax - min_ax;
  float span_ay = max_ay - min_ay;
  float span_az = max_az - min_az;

  if (span_ax < 2.0f || span_ay < 2.0f || span_az < 2.0f) {
    Serial.println("# ОШИБКА: слишком малый размах по одной из осей. Проверьте положения (ожидается ~2g).");
    return false;
  }

  acc_bias_x = (max_ax + min_ax) * 0.5f;
  acc_bias_y = (max_ay + min_ay) * 0.5f;
  acc_bias_z = (max_az + min_az) * 0.5f;
  acc_scale_x = span_ax / (2.0f * G_MS2);
  acc_scale_y = span_ay / (2.0f * G_MS2);
  acc_scale_z = span_az / (2.0f * G_MS2);

  Serial.println();
  Serial.println("# ---------- ИТОГ КАЛИБРОВКИ АКСЕЛЕРОМЕТРА (ось датчика, для подстановки в код) ----------");
  Serial.println("# Сырые средние по позам (м/с²): индекс 0..5 как порядок поз выше.");
  for (uint8_t i = 0; i < 6; i++) {
    Serial.print("# [");
    Serial.print(i);
    Serial.print("] ax=");
    Serial.print(pose_mean[i].x, 5);
    Serial.print(" ay=");
    Serial.print(pose_mean[i].y, 5);
    Serial.print(" az=");
    Serial.println(pose_mean[i].z, 5);
  }
  Serial.print("# const float ACC_BX = ");
  Serial.print(acc_bias_x, 6);
  Serial.println("f;");
  Serial.print("# const float ACC_BY = ");
  Serial.print(acc_bias_y, 6);
  Serial.println("f;");
  Serial.print("# const float ACC_BZ = ");
  Serial.print(acc_bias_z, 6);
  Serial.println("f;");
  Serial.print("# const float ACC_SX = ");
  Serial.print(acc_scale_x, 6);
  Serial.println("f;");
  Serial.print("# const float ACC_SY = ");
  Serial.print(acc_scale_y, 6);
  Serial.println("f;");
  Serial.print("# const float ACC_SZ = ");
  Serial.print(acc_scale_z, 6);
  Serial.println("f;");
  Serial.println("# ------------------------------------------------------------------------------");
  return true;
}

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 4000)
    ;

  imuOk = mpu.begin();
  if (!imuOk) {
    Serial.println("# MPU6050 не найден. Проверьте I2C.");
    return;
  }

  mpu.setAccelerometerRange(MPU6050_RANGE_4_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);

  phase = PHASE_CAL_ACCEL;
  accel_pose_idx = 0;
  accel_sub = ACCEL_WAIT_ENTER;
  flush_serial_input();

  print_banner("# MPU6050 OK.");
  Serial.print("# Усреднение каждой позы: ");
  Serial.print(POSE_COLLECT_MS / 1000UL);
  Serial.println(" с (измените POSE_COLLECT_MS в скетче при необходимости).");
  Serial.println(POSE_HINT(0));
  Serial.println("# Установите плату и отправьте любой символ (например Enter), чтобы начать сбор этой позы.");
}

void loop() {
  if (!imuOk)
    return;

  if (phase == PHASE_DONE)
    return;

  unsigned long now_ms = millis();

  sensors_event_t accel, gyro, temp;
  mpu.getEvent(&accel, &gyro, &temp);

  float ax_raw = accel.acceleration.x;
  float ay_raw = accel.acceleration.y;
  float az_raw = accel.acceleration.z;

  // ========== Фаза A: 6 поз акселя ==========
  if (phase == PHASE_CAL_ACCEL) {
    if (accel_sub == ACCEL_WAIT_ENTER) {
      if (Serial.available() > 0) {
        flush_serial_input();
        accel_sub = ACCEL_COLLECTING;
        accel_pose_deadline_ms = now_ms + POSE_COLLECT_MS;
        accel_sum_ax = accel_sum_ay = accel_sum_az = 0.0f;
        accel_pose_samples = 0;
        Serial.println("# Сбор начался — держите НЕПОДВИЖНО до сообщения «Поза сохранена».");
      }
      return;
    }

    accel_sum_ax += ax_raw;
    accel_sum_ay += ay_raw;
    accel_sum_az += az_raw;
    accel_pose_samples++;

    if (now_ms >= accel_pose_deadline_ms) {
      if (accel_pose_samples == 0) {
        Serial.println("# Ошибка: нет выборок.");
        return;
      }
      pose_mean[accel_pose_idx].x = accel_sum_ax / (float)accel_pose_samples;
      pose_mean[accel_pose_idx].y = accel_sum_ay / (float)accel_pose_samples;
      pose_mean[accel_pose_idx].z = accel_sum_az / (float)accel_pose_samples;

      Serial.print("# Поза сохранена ");
      Serial.print(accel_pose_idx + 1);
      Serial.print("/6, выборок: ");
      Serial.println(accel_pose_samples);

      accel_pose_idx++;
      if (accel_pose_idx >= 6) {
        if (!compute_accel_calibration_from_poses()) {
          phase = PHASE_DONE;
          Serial.println("# Остановка: исправьте калибровку поз и перезагрузите плату.");
          return;
        }
        phase = PHASE_CAL_GYRO;
        phaseStartMs = now_ms;
        sum_gx_raw = sum_gy_raw = sum_gz_raw = 0.0f;
        cal_count = 0;
        print_banner("# Положите плату ГОРИЗОНТАЛЬНО и НЕ ДВИГАЙТЕ ~5 с — калибровка гироскопа (реальное усреднение).");
      } else {
        accel_sub = ACCEL_WAIT_ENTER;
        Serial.println();
        Serial.println(POSE_HINT(accel_pose_idx));
        Serial.println("# Отправьте символ, когда готовы начать сбор следующей позы.");
      }
    }
    return;
  }

  // Дальше нужны углы и body-ускорение — считаем после калибровки акселя
  float ax_b = 0.0f, ay_b = 0.0f, az_b = 0.0f;
  body_from_sensor_cal(ax_raw, ay_raw, az_raw, ax_b, ay_b, az_b);

  float gx_sensor = (gyro.gyro.x * 180.0f / PI) - cal_bx;
  float gy_sensor = (gyro.gyro.y * 180.0f / PI) - cal_by;
  float gz_sensor = (gyro.gyro.z * 180.0f / PI) - cal_bz;
  float gx = gz_sensor;
  float gy = -gy_sensor;
  float gz = gx_sensor;

  unsigned long now_us = micros();
  float dt_sec = (lastAngleUs != 0UL) ? ((now_us - lastAngleUs) / 1000000.0f) : 0.001f;
  lastAngleUs = now_us;
  if (dt_sec <= 0.0f || dt_sec > 0.25f)
    dt_sec = 0.001f;

  // ========== Фаза B: гиро ==========
  if (phase == PHASE_CAL_GYRO) {
    sum_gx_raw += gyro.gyro.x * (180.0f / PI);
    sum_gy_raw += gyro.gyro.y * (180.0f / PI);
    sum_gz_raw += gyro.gyro.z * (180.0f / PI);
    cal_count++;

    if (now_ms - phaseStartMs >= CALIBRATION_DURATION_MS) {
      if (cal_count > 0) {
        cal_bx = sum_gx_raw / (float)cal_count;
        cal_by = sum_gy_raw / (float)cal_count;
        cal_bz = sum_gz_raw / (float)cal_count;
      }
      Serial.println();
      Serial.println("# ---------- ИТОГ КАЛИБРОВКИ ГИРОСКОПА (°/с, сырое усреднение по датчику) ----------");
      Serial.print("# cal_bx = ");
      Serial.print(cal_bx, 6);
      Serial.println("f;");
      Serial.print("# cal_by = ");
      Serial.print(cal_by, 6);
      Serial.println("f;");
      Serial.print("# cal_bz = ");
      Serial.print(cal_bz, 6);
      Serial.println("f;");
      Serial.println("# В прошивке arduino_ins_gnss можно временно подставить их как начальные bx,by,bz.");
      Serial.println("# ------------------------------------------------------------------------------");

      phase = PHASE_SETTLE;
      phaseStartMs = now_ms;
      lastAngleUs = micros();
      roll_deg = pitch_deg = yaw_deg = 0.0f;
      roll_prev = pitch_prev = 0.0f;
      print_banner("# Прогрев оценки roll/pitch ~3 с (без статистики дисперсии).");
    }
    return;
  }

  float roll_accel = 0.0f, pitch_accel = 0.0f;
  float den_yz = sqrtf(ay_b * ay_b + az_b * az_b);
  if (den_yz > 1e-6f)
    roll_accel = atan2f(ay_b, az_b) * (180.0f / PI);
  if (den_yz > 1e-6f)
    pitch_accel = atan2f(-ax_b, den_yz) * (180.0f / PI);

  float roll_gyro = roll_prev + gx * dt_sec;
  float pitch_gyro = pitch_prev + gy * dt_sec;
  roll_deg = COMPL_ALPHA * roll_gyro + (1.0f - COMPL_ALPHA) * roll_accel;
  pitch_deg = COMPL_ALPHA * pitch_gyro + (1.0f - COMPL_ALPHA) * pitch_accel;
  roll_prev = roll_deg;
  pitch_prev = pitch_deg;

  yaw_deg += gz * dt_sec;
  if (yaw_deg > 180.0f)
    yaw_deg -= 360.0f;
  if (yaw_deg < -180.0f)
    yaw_deg += 360.0f;

  if (phase == PHASE_SETTLE) {
    if (now_ms - phaseStartMs >= SETTLE_DURATION_MS) {
      phase = PHASE_COLLECT;
      phaseStartMs = now_ms;
      lastAngleUs = micros();
      lastProgressMs = phaseStartMs;
      mean_ax_h = mean_ay_h = 0.0f;
      m2_ax_h = m2_ay_h = 0.0f;
      n_stats = 0;
      print_banner("# Сбор дисперсии ah_x, ah_y — 10 минут. Плату не двигать.");
    }
    return;
  }

  if (phase == PHASE_COLLECT) {
    float ah_x = 0.0f, ah_y = 0.0f, ah_z_unused = 0.0f;
    accel_body_to_level(ax_b, ay_b, az_b, roll_deg, pitch_deg, ah_x, ah_y, ah_z_unused);

    welford_update_pair(ah_x, ah_y, mean_ax_h, mean_ay_h, m2_ax_h, m2_ay_h, n_stats);

    if (now_ms - lastProgressMs >= 60000UL) {
      lastProgressMs += 60000UL;
      Serial.print("# Прогресс сбора: ");
      Serial.print((now_ms - phaseStartMs) / 60000UL);
      Serial.println(" мин из 10");
    }

    if (now_ms - phaseStartMs >= COLLECT_DURATION_MS) {
      phase = PHASE_DONE;
      float var_x = (n_stats > 1UL) ? (m2_ax_h / (float)(n_stats - 1UL)) : 0.0f;
      float var_y = (n_stats > 1UL) ? (m2_ay_h / (float)(n_stats - 1UL)) : 0.0f;

      Serial.println();
      Serial.println("# ========== ИТОГ ДИСПЕРСИИ (горизонталь после выставки roll/pitch) ==========");
      Serial.print("# Выборок: ");
      Serial.println(n_stats);
      Serial.print("# Среднее ah_x (м/с²): ");
      Serial.println(mean_ax_h, 6);
      Serial.print("# Среднее ah_y (м/с²): ");
      Serial.println(mean_ay_h, 6);
      Serial.print("# Дисперсия ah_x (м/с²)²: ");
      Serial.println(var_x, 8);
      Serial.print("# Дисперсия ah_y (м/с²)²: ");
      Serial.println(var_y, 8);
      Serial.print("# СКО ah_x (м/с²): ");
      Serial.println(sqrtf(var_x), 6);
      Serial.print("# СКО ah_y (м/с²): ");
      Serial.println(sqrtf(var_y), 6);
      Serial.println("# ================================================================================");
      Serial.println("# --- Фильтр Калмана: процессный шум Q (кратко) ---");
      Serial.println("# var_x = sigma_ax_horiz^2, var_y = sigma_ay_horiz^2  [(м/с^2)^2]");
      Serial.println("# Это оценки ШУМА акселерометра в горизонтальной плоскости после выставки по крену/тангажу.");
      Serial.println("# R задаёт неопределённость ИЗМЕРЕНИЙ (ГНСС и т.д.); Q — неопределённость ЭВОЛЮЦИИ состояния");
      Serial.println("# при распространении модели (типично — вклад ИМУ). Не подменяйте Q матрицей R.");
      Serial.println("# Если в модели шум входит как случайное ускорение w_a и состояние содержит скорость с шагом dt,");
      Serial.println("# часто используют порядок Q_vel ~ sigma_a^2 * dt^2 (константы множителя берите из вашего разложения F, G).");
      Serial.println("# Для второй горизонтальной оси — аналогично второй диагонали блока ускорения/скорости.");
      Serial.println("# Отдельно в Q могут понадобиться члены от ДРЕЙФА ГИРО (этот скетч их не оценивает).");
      Serial.println("# ================================================================================");
      Serial.println("# Готово. Перезагрузите плату для нового цикла.");
    }
  }
}
