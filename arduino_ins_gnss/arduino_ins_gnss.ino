/*
 * Сбор данных ИНС (IMU) и ГНСС для последующего комплексирования в Python.
 *
 * Оборудование:
 *   - ГНСС: NEO-6M / NEO-M8N по UART (SoftwareSerial на пинах 4-RX, 5-TX)
 *   - ИНС: MPU6050 по I2C (SDA, SCL)
 *
 * Выход: CSV в Serial (115200). Чтобы сохранить в файл на ПК — запустите
 *        serial_to_file.py и подключите Arduino к COM-порту.
 * Колонки: time_ms,lat,lon,alt_m,speed_mps,sats,fix,ax,ay,az,gx,gy,gz,roll_deg,pitch_deg,yaw_deg
 * ax,ay,az — ускорение в СК body (м/с²). gx,gy,gz — угловые скорости в СК body (град/с).
 * roll_deg, pitch_deg — крен/тангаж: комплементарный фильтр. yaw_deg — интеграл gz.
 * При старте: автокалибровка дрейфа гиро (держите неподвижно ~5 с), затем «Начинайте движение».
 */

#include <Wire.h>
#include <TinyGPSPlus.h>
#include <SoftwareSerial.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

#define GNSS_RX_PIN  4
#define GNSS_TX_PIN  5
#define GNSS_BAUD    9600

SoftwareSerial gnssSerial(GNSS_RX_PIN, GNSS_TX_PIN);
TinyGPSPlus    gps;
Adafruit_MPU6050 mpu;

const unsigned long OUTPUT_INTERVAL_MS = 100;  // 10 Гц
unsigned long lastOutputMs = 0;

// Углы курсовертикали: комплементарный фильтр (крен/тангаж), рысканье — интеграл gz
float roll_deg = 0.0f, pitch_deg = 0.0f, yaw_deg = 0.0f;
float roll_prev = 0.0f, pitch_prev = 0.0f;  // для комплементарного фильтра
unsigned long lastAngleMs = 0;
const float COMPL_ALPHA = 0.98f;  // доля гиро в комплементарном фильтре (0.98 = меньше дрейф от акселерометра)

// --- 6-pose calibration (sensor frame), then SENSOR -> BODY ---
const float G = 9.80665f;
const float ACC_BX =  0.459136f;   // m/s^2
const float ACC_BY = -0.249651f;
const float ACC_BZ = -0.753802f;
const float ACC_SX =  0.993740f;   // unitless
const float ACC_SY =  1.004900f;
const float ACC_SZ =  1.017190f;

// Смещение нуля гиро: после автокалибровки подставляются cal_bx, cal_by, cal_bz
const float bx = -5.362898;
const float by = 0.597485;
const float bz = -0.228809;
float cal_bx = bx, cal_by = by, cal_bz = bz;  // используются после калибровки

// Автокалибровка дрейфа гиро (в покое)
const unsigned long CALIBRATION_DURATION_MS = 5000;
bool calibrating = true;
unsigned long calibrationStartMs = 0;
float sum_gx = 0.0f, sum_gy = 0.0f, sum_gz = 0.0f;
int cal_count = 0;
unsigned long lastCalMsgMs = 0;

bool imuOk = false;

void setup() {
  Serial.begin(115200);
  gnssSerial.begin(GNSS_BAUD);

  if (!mpu.begin()) {
    Serial.println("# MPU6050 не найден. Проверьте подключение I2C.");
  } else {
    mpu.setAccelerometerRange(MPU6050_RANGE_4_G);
    mpu.setGyroRange(MPU6050_RANGE_500_DEG);
    mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
    imuOk = true;
    Serial.println("# MPU6050 OK");
    calibrationStartMs = millis();
    sum_gx = 0.0f;
    sum_gy = 0.0f;
    sum_gz = 0.0f;
    cal_count = 0;
  }

  Serial.println("# Ожидание фиксации ГНСС...");
  Serial.println("time_ms,lat,lon,alt_m,speed_mps,sats,fix,ax,ay,az,gx,gy,gz,roll_deg,pitch_deg,yaw_deg");
  lastOutputMs = millis();
}

void loop() {
  while (gnssSerial.available() > 0) {
    gps.encode(gnssSerial.read());
  }

  unsigned long now = millis();

  // Фаза автокалибровки гиро: держите плату неподвижно
  if (imuOk && calibrating) {
    unsigned long elapsed = now - calibrationStartMs;
    if (elapsed >= CALIBRATION_DURATION_MS) {
      if (cal_count > 0) {
        cal_bx = sum_gx / (float)cal_count;
        cal_by = sum_gy / (float)cal_count;
        cal_bz = sum_gz / (float)cal_count;
      }
      calibrating = false;
      Serial.println("# Калибровка завершена. Начинайте движение.");
    } else {
      sensors_event_t accel, gyro, temp;
      mpu.getEvent(&accel, &gyro, &temp);
      if (cal_count == 0)
        Serial.println("# Держите неподвижно. Калибровка гиро... 5 сек");
      sum_gx += gyro.gyro.x * (180.0f / PI);
      sum_gy += gyro.gyro.y * (180.0f / PI);
      sum_gz += gyro.gyro.z * (180.0f / PI);
      cal_count++;
      if (now - lastCalMsgMs >= 1000) {
        lastCalMsgMs = now;
        Serial.print("# Держите неподвижно. Калибровка гиро... ");
        Serial.print((CALIBRATION_DURATION_MS - elapsed) / 1000);
        Serial.println(" сек");
      }
    }
    return;
  }

  if (now - lastOutputMs < OUTPUT_INTERVAL_MS)
    return;
  lastOutputMs = now;

  float lat   = gps.location.isValid()   ? (float)gps.location.lat()   : NAN;
  float lon   = gps.location.isValid()   ? (float)gps.location.lng()   : NAN;
  float alt   = gps.altitude.isValid()   ? (float)gps.altitude.meters() : NAN;
  float speed = gps.speed.isValid()      ? (float)gps.speed.mps()      : NAN;
  int   sats  = gps.satellites.isValid() ? (int)gps.satellites.value() : 0;
  int   fix   = gps.location.isValid()   ? 1 : 0;

  float ax = NAN, ay = NAN, az = NAN, gx = NAN, gy = NAN, gz = NAN;
  if (imuOk) {
    sensors_event_t accel, gyro, temp;
    mpu.getEvent(&accel, &gyro, &temp);
    // 1) Calibrate in SENSOR frame (bias + scale)
    float ax_raw = accel.acceleration.x;
    float ay_raw = accel.acceleration.y;
    float az_raw = accel.acceleration.z;
    float ax_cal = (ax_raw - ACC_BX) / ACC_SX;
    float ay_cal = (ay_raw - ACC_BY) / ACC_SY;
    float az_cal = (az_raw - ACC_BZ) / ACC_SZ;
    // 2) Map SENSOR -> BODY (одинаково для акселерометра и гироскопа)
    ax = az_cal;
    ay = -ay_cal;
    az = ax_cal;
    float gx_sensor = (gyro.gyro.x * 180.0f / PI) - cal_bx;
    float gy_sensor = (gyro.gyro.y * 180.0f / PI) - cal_by;
    float gz_sensor = (gyro.gyro.z * 180.0f / PI) - cal_bz;
    gx = gz_sensor;   // body X <- sensor Z
    gy = -gy_sensor;  // body Y <- -sensor Y
    gz = gx_sensor;   // body Z <- sensor X (ось рысканья)

    float dt_sec = (lastAngleMs != 0) ? ((now - lastAngleMs) / 1000.0f) : 0.01f;
    if (dt_sec <= 0.0f || dt_sec > 1.0f) dt_sec = 0.01f;
    lastAngleMs = now;

    // Углы по акселерометру (верны только при |a| ≈ g)
    float roll_accel = 0.0f, pitch_accel = 0.0f;
    float den_yz = sqrtf(ay * ay + az * az);
    if (den_yz > 1e-6f)
      roll_accel = atan2f(ay, az) * (180.0f / PI);
    float den_x = sqrtf(ay * ay + az * az);
    if (den_x > 1e-6f)
      pitch_accel = atan2f(-ax, den_x) * (180.0f / PI);

    // Комплементарный фильтр: крен и тангаж = alpha*(гиро) + (1-alpha)*(акселерометр)
    float roll_gyro = roll_prev + gx * dt_sec;
    float pitch_gyro = pitch_prev + gy * dt_sec;
    roll_deg  = COMPL_ALPHA * roll_gyro  + (1.0f - COMPL_ALPHA) * roll_accel;
    pitch_deg = COMPL_ALPHA * pitch_gyro + (1.0f - COMPL_ALPHA) * pitch_accel;
    roll_prev = roll_deg;
    pitch_prev = pitch_deg;

    // Рысканье: интеграл угловой скорости по body Z
    yaw_deg += gz * dt_sec;
    if (yaw_deg > 180.0f)  yaw_deg -= 360.0f;
    if (yaw_deg < -180.0f) yaw_deg += 360.0f;
  }

  // CSV в Serial — сохраняется на ПК через serial_to_file.py
  Serial.print(now);
  Serial.print(",");
  printFloat(lat);
  Serial.print(",");
  printFloat(lon);
  Serial.print(",");
  printFloat(alt);
  Serial.print(",");
  printFloat(speed);
  Serial.print(",");
  Serial.print(sats);
  Serial.print(",");
  Serial.print(fix);
  Serial.print(",");
  printFloat(ax);
  Serial.print(",");
  printFloat(ay);
  Serial.print(",");
  printFloat(az);
  Serial.print(",");
  printFloat(gx);
  Serial.print(",");
  printFloat(gy);
  Serial.print(",");
  printFloat(gz);
  Serial.print(",");
  if (imuOk) {
    printFloat(roll_deg);
    Serial.print(",");
    printFloat(pitch_deg);
    Serial.print(",");
    printFloat(yaw_deg);
  } else {
    Serial.print("nan,nan,nan");
  }
  Serial.println();

  if (gps.charsProcessed() < 10 && now > 5000) {
    Serial.println("# Внимание: нет данных ГНСС. Проверьте питание и RX/TX.");
  }
}

void printFloat(float x) {
  if (isnan(x)) {
    Serial.print("nan");
    return;
  }
  Serial.print(x, 6);
}
