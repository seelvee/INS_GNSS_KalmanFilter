/*
 * Сбор датасета ИНС (MPU6050 DMP) + ГНСС.
 *
 * Оборудование:
 *   - ГНСС: NEO-6M / NEO-M8N по UART
 *       * Mega2560: аппаратный Serial1 (RX1=19, TX1=18)
 *       * Uno/Nano: SoftwareSerial (RX=4, TX=5)
 *   - ИНС: MPU6050 по I2C. Рекомендуется: вывод INT датчика на пин 2 (DMP data ready).
 *
 * Библиотеки: https://github.com/jrowberg/i2cdevlib
 *   Склонировать репозиторий, в Arduino скопировать в libraries/ папки:
 *   - i2cdevlib/Arduino/I2Cdev  -> I2Cdev
 *   - i2cdevlib/Arduino/MPU6050  -> MPU6050
 *   TinyGPSPlus — из Library Manager.
 *
 * Выход: CSV в Serial (115200).
 * Колонки (минимум для check_map.py):
 * time_ms,lat,lon,ax,ay,az,roll_dmp_deg,pitch_dmp_deg,yaw_dmp_deg
 *
 * Важно:
 * - ax..az — ускорение из DMP/FIFO, пересчитанное в м/с^2 (FS=±2g, 16384 LSB/g).
 * - roll/pitch/yaw — оценка ориентации от DMP.
 */

#include <Wire.h>
#include <TinyGPSPlus.h>
#include <SoftwareSerial.h>

#include "I2Cdev.h"
#include "MPU6050_6Axis_MotionApps20.h"

#define GNSS_BAUD    9600
#define INTERRUPT_PIN 2   // INT MPU6050 -> пин 2 (можно не подключать, тогда опрос по таймеру)

TinyGPSPlus    gps;
MPU6050        mpu;

#if defined(ARDUINO_AVR_MEGA2560)
  #define GNSS_SERIAL Serial1
#else
  #define GNSS_RX_PIN  4
  #define GNSS_TX_PIN  5
  SoftwareSerial gnssSerial(GNSS_RX_PIN, GNSS_TX_PIN);
  #define GNSS_SERIAL gnssSerial
#endif



const unsigned long OUTPUT_INTERVAL_MS = 100;
unsigned long lastOutputMs = 0;
const unsigned long GNSS_HEADING_CALIB_INTERVAL_MS = 30000;
const float GNSS_MIN_SPEED_FOR_COURSE_MPS = 0.7f;
const float GRAVITY_MPS2 = 9.80665f;
const float ACC_SCALE_RAW_TO_MPS2 = (GRAVITY_MPS2 / 16384.0f) * 2;  // MPU6050 FS=±2g
const float GYRO_SCALE_RAW_TO_DPS = 2.0f / 131.0f;  // MPU6050 FS=±250 dps

// DMP
bool dmpReady = false;
uint8_t devStatus;
uint16_t packetSize;
uint8_t fifoBuffer[64];
Quaternion q;
VectorInt16 aa;
VectorFloat gravity;
float ypr[3];  // yaw, pitch, roll в радианах

// Автокалибровка гиро перед включением DMP
const unsigned long CALIBRATION_DURATION_MS = 5000;
int32_t sum_gx = 0, sum_gy = 0, sum_gz = 0;
int cal_count = 0;

float roll_deg = NAN, pitch_deg = NAN, yaw_deg = NAN;
int16_t ax_raw = 0, ay_raw = 0, az_raw = 0;
int16_t gx_raw = 0, gy_raw = 0, gz_raw = 0;
float ax_mps2 = NAN, ay_mps2 = NAN, az_mps2 = NAN;
float gx_dps = NAN, gy_dps = NAN, gz_dps = NAN;
float heading_est_deg = NAN;
bool heading_initialized = false;
unsigned long lastHeadingUpdateMs = 0;
unsigned long lastHeadingCalibMs = 0;

float wrapAngle360(float deg) {
  while (deg < 0.0f) deg += 360.0f;
  while (deg >= 360.0f) deg -= 360.0f;
  return deg;
}

void setup() {
  Serial.begin(115200);
  GNSS_SERIAL.begin(GNSS_BAUD);
  Wire.begin();
  Wire.setClock(400000);

  Serial.println("# Инициализация MPU6050...");
  mpu.initialize();
  pinMode(INTERRUPT_PIN, INPUT);

  if (!mpu.testConnection()) {
    Serial.println("# MPU6050 не найден. Проверьте I2C.");
    return;
  }
  Serial.println("# MPU6050 OK");
  // Явно фиксируем диапазоны, чтобы шкалы raw в Python были однозначными.
  mpu.setFullScaleAccelRange(MPU6050_ACCEL_FS_2);
  mpu.setFullScaleGyroRange(MPU6050_GYRO_FS_250);

  // Калибровка гиро в покое (5 сек)
  Serial.println("# Держите неподвижно. Калибровка гиро... 5 сек");
  unsigned long t0 = millis();
  unsigned long lastMsg = 0;
  while (millis() - t0 < CALIBRATION_DURATION_MS) {
    sum_gx += mpu.getRotationX();
    sum_gy += mpu.getRotationY();
    sum_gz += mpu.getRotationZ();
    cal_count++;
    if (millis() - lastMsg >= 1000) {
      lastMsg = millis();
      Serial.print("# Держите неподвижно... ");
      Serial.print((CALIBRATION_DURATION_MS - (millis() - t0)) / 1000);
      Serial.println(" сек");
    }
    delay(2);
  }
  if (cal_count > 0) {
    mpu.setXGyroOffset((int16_t)(-sum_gx / cal_count));
    mpu.setYGyroOffset((int16_t)(-sum_gy / cal_count));
    mpu.setZGyroOffset((int16_t)(-sum_gz / cal_count));
  }
  Serial.println("# Калибровка гиро завершена.");

  // Загрузка и включение DMP
  devStatus = mpu.dmpInitialize();
  if (devStatus != 0) {
    Serial.print("# Ошибка DMP (код ");
    Serial.print(devStatus);
    Serial.println("). Углы будут 0.");
  } else {
    mpu.setDMPEnabled(true);
    attachInterrupt(digitalPinToInterrupt(INTERRUPT_PIN), dmpDataReady, RISING);
    packetSize = mpu.dmpGetFIFOPacketSize();
    dmpReady = true;
    Serial.println("# DMP готов. Начинайте движение.");
  }

  Serial.println("# Ожидание фиксации ГНСС...");
  Serial.println("time_ms,lat,lon,gnss_speed_mps,gnss_vn_mps,gnss_ve_mps,ax,ay,az,gx,gy,gz,roll_deg,pitch_deg,yaw_deg");
  lastOutputMs = millis();
}

volatile bool mpuInterrupt = false;
void dmpDataReady() {
  mpuInterrupt = true;
}

void loop() {
  while (GNSS_SERIAL.available() > 0)
    gps.encode(GNSS_SERIAL.read());

  unsigned long now = millis();
  if (now - lastOutputMs < OUTPUT_INTERVAL_MS)
    return;
  unsigned long dtMs = now - lastOutputMs;
  lastOutputMs = now;

  float lat   = gps.location.isValid()   ? (float)gps.location.lat()   : NAN;
  float lon   = gps.location.isValid()   ? (float)gps.location.lng()   : NAN;
  float gnss_speed_mps = gps.speed.isValid() ? (float)gps.speed.mps() : NAN;
  float gnss_course_deg = gps.course.isValid() ? (float)gps.course.deg() : NAN;
  float gnss_course_rad = !isnan(gnss_course_deg) ? (float)(gnss_course_deg * (PI / 180.0f)) : NAN;
  float gnss_vn_mps = (!isnan(gnss_speed_mps) && !isnan(gnss_course_rad))
                        ? gnss_speed_mps * cos(gnss_course_rad)
                        : NAN;
  float gnss_ve_mps = (!isnan(gnss_speed_mps) && !isnan(gnss_course_rad))
                        ? gnss_speed_mps * sin(gnss_course_rad)
                        : NAN;
  ax_raw = 0; ay_raw = 0; az_raw = 0;
  gx_raw = 0; gy_raw = 0; gz_raw = 0;
  ax_mps2 = NAN; ay_mps2 = NAN; az_mps2 = NAN;
  gx_dps = NAN; gy_dps = NAN; gz_dps = NAN;
  roll_deg = NAN; pitch_deg = NAN; yaw_deg = NAN;

  if (dmpReady && mpu.dmpGetCurrentFIFOPacket(fifoBuffer)) {
    mpu.dmpGetQuaternion(&q, fifoBuffer);
    mpu.dmpGetGravity(&gravity, &q);
    mpu.dmpGetYawPitchRoll(ypr, &q, &gravity);

    // выдает результат действительно в радианах  
    roll_deg = ypr[2] * (180.0f / PI);
    pitch_deg = ypr[1] * (180.0f / PI);
    yaw_deg = ypr[0] * (180.0f / PI);

    mpu.dmpGetAccel(&aa, fifoBuffer);
    ax_raw = aa.x;
    ay_raw = aa.y;
    az_raw = aa.z;
    ax_mps2 = ax_raw * ACC_SCALE_RAW_TO_MPS2;
    ay_mps2 = ay_raw * ACC_SCALE_RAW_TO_MPS2;
    az_mps2 = az_raw * ACC_SCALE_RAW_TO_MPS2;

    mpu.dmpGetGyro(&aa, fifoBuffer);
    gx_raw = aa.x;
    gy_raw = aa.y;
    gz_raw = aa.z;
    gx_dps = gx_raw * GYRO_SCALE_RAW_TO_DPS;
    gy_dps = gy_raw * GYRO_SCALE_RAW_TO_DPS;
    gz_dps = gz_raw * GYRO_SCALE_RAW_TO_DPS;
  }

  const bool gnssCourseUsable =
    !isnan(gnss_course_deg) &&
    !isnan(gnss_speed_mps) &&
    (gnss_speed_mps >= GNSS_MIN_SPEED_FOR_COURSE_MPS);

  if (!heading_initialized && gnssCourseUsable) {
    heading_est_deg = wrapAngle360(gnss_course_deg);
    heading_initialized = true;
    lastHeadingUpdateMs = now;
    lastHeadingCalibMs = now;
  }

  if (heading_initialized && !isnan(gz_dps)) {
    if (lastHeadingUpdateMs != 0 && now > lastHeadingUpdateMs) {
      float dt_s = (float)(now - lastHeadingUpdateMs) * 0.001f;
      heading_est_deg = wrapAngle360(heading_est_deg + gz_dps * dt_s);
    }
    lastHeadingUpdateMs = now;
  }

  if (heading_initialized && gnssCourseUsable && (now - lastHeadingCalibMs >= GNSS_HEADING_CALIB_INTERVAL_MS)) {
    heading_est_deg = wrapAngle360(gnss_course_deg);
    lastHeadingCalibMs = now;
  }

  if (heading_initialized) {
    yaw_deg = heading_est_deg;
  } else {
    yaw_deg = NAN;
  }

  Serial.print(now);
  Serial.print(",");
  printFloat(lat);
  Serial.print(",");
  printFloat(lon);
  Serial.print(",");
  printFloat(gnss_speed_mps);
  Serial.print(",");
  printFloat(gnss_vn_mps);
  Serial.print(",");
  printFloat(gnss_ve_mps);
  Serial.print(",");
  printFloat(ax_mps2);
  Serial.print(",");
  printFloat(ay_mps2);
  Serial.print(",");
  printFloat(az_mps2);
  Serial.print(",");
  printFloat(gx_dps);
  Serial.print(",");
  printFloat(gy_dps);
  Serial.print(",");
  printFloat(gz_dps);
  Serial.print(",");
  printFloat(roll_deg);
  Serial.print(",");
  printFloat(pitch_deg);
  Serial.print(",");
  printFloat(yaw_deg);

  Serial.println();

  if (gps.charsProcessed() < 10 && now > 5000)
    Serial.println("# Внимание: нет данных ГНСС.");
}

void printFloat(float x) {
  if (isnan(x)) { Serial.print("nan"); return; }
  Serial.print(x, 6);
}
