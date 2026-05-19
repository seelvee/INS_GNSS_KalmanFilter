/*
 * MPU6050 — поток сырых данных по Serial (CSV) + простые команды.
 * Плата: Arduino Uno, I2C: SDA=A4, SCL=A5. Serial: 115200 бод.
 *
 * Режим ±2g / ±250 °/с, выборка ~100 Гц (SMPLRT_DIV + DLPF).
 * Формат строки потока:
 *   time_ms,ax_raw,ay_raw,az_raw,gx_raw,gy_raw,gz_raw,temp_raw
 *
 * Команды (строка в Serial, без учёта регистра, завершение \\r или \\n):
 *   START_STREAM  — начать вывод CSV
 *   STOP_STREAM   — остановить вывод
 *   WHOAMI        — прочитать регистр WHO_AM_I (0x75)
 *   RESET         — аппаратный сброс MPU6050 и повторная инициализация
 */

#include <Wire.h>

static const uint8_t MPU6050_ADDR = 0x68;

// Регистры
static const uint8_t RA_PWR_MGMT_1 = 0x6B;
static const uint8_t RA_SMPLRT_DIV = 0x19;
static const uint8_t RA_CONFIG = 0x1A;
static const uint8_t RA_GYRO_CONFIG = 0x1B;
static const uint8_t RA_ACCEL_CONFIG = 0x1C;
static const uint8_t RA_ACCEL_XOUT_H = 0x3B;
static const uint8_t RA_TEMP_OUT_H = 0x41;
static const uint8_t RA_WHO_AM_I = 0x75;

static const uint16_t STREAM_INTERVAL_MS = 10;  // ~100 Гц

bool mpuPresent = false;
bool streaming = false;
unsigned long lastStreamMs = 0;

String cmdBuf;

static uint8_t readByte(uint8_t reg, bool &ok) {
  ok = false;
  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0)
    return 0;
  uint8_t n = Wire.requestFrom((uint8_t)MPU6050_ADDR, (uint8_t)1);
  if (n != 1)
    return 0;
  ok = true;
  return Wire.read();
}

static bool writeByte(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(reg);
  Wire.write(val);
  return Wire.endTransmission(true) == 0;
}

static bool readBurst(uint8_t reg, uint8_t *buf, uint8_t len) {
  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0)
    return false;
  uint8_t got = Wire.requestFrom((uint8_t)MPU6050_ADDR, len);
  if (got != len)
    return false;
  for (uint8_t i = 0; i < len; i++)
    buf[i] = Wire.read();
  return true;
}

static int16_t be16(const uint8_t *p) {
  return (int16_t)(((uint16_t)p[0] << 8) | (uint16_t)p[1]);
}

static bool pingMpu6050() {
  bool ok = false;
  uint8_t v = readByte(RA_WHO_AM_I, ok);
  return ok && v == 0x68;
}

static bool initMpu6050() {
  if (!writeByte(RA_PWR_MGMT_1, 0x80)) {  // DEVICE_RESET
    Serial.println("# ERROR: не удалось записать PWR_MGMT_1 (RESET)");
    return false;
  }
  delay(150);

  // Выход из сна, выбор оси времени гироскопа по умолчанию
  if (!writeByte(RA_PWR_MGMT_1, 0x00)) {
    Serial.println("# ERROR: не удалось вывести MPU6050 из сна");
    return false;
  }
  delay(10);

  // DLPF включён: для большинства режимов частота гиро ~1 кГц → SMPLRT_DIV=9 → ~100 Гц
  if (!writeByte(RA_CONFIG, 0x03)) {  // DLPF ~44 Hz band (typ. gyro rate 1 kHz)
    Serial.println("# ERROR: CONFIG");
    return false;
  }
  if (!writeByte(RA_SMPLRT_DIV, 9)) {
    Serial.println("# ERROR: SMPLRT_DIV");
    return false;
  }

  // ±250 °/с: FS_SEL = 0
  if (!writeByte(RA_GYRO_CONFIG, 0x00)) {
    Serial.println("# ERROR: GYRO_CONFIG");
    return false;
  }

  // ±2g: AFS_SEL = 0
  if (!writeByte(RA_ACCEL_CONFIG, 0x00)) {
    Serial.println("# ERROR: ACCEL_CONFIG");
    return false;
  }

  delay(10);

  if (!pingMpu6050()) {
    Serial.println("# ERROR: MPU6050 не отвечает после инициализации (WHO_AM_I != 0x68)");
    return false;
  }

  Serial.println("# MPU6050 OK: ±2g, ±250 °/с, ~100 Гц. Отправьте START_STREAM для потока CSV.");
  return true;
}

static void handleWhoAmI() {
  bool ok = false;
  uint8_t v = readByte(RA_WHO_AM_I, ok);
  if (!ok) {
    Serial.println("# WHOAMI: ошибка чтения I2C");
    return;
  }
  Serial.print("# WHO_AM_I = 0x");
  if (v < 16)
    Serial.print('0');
  Serial.println(v, HEX);
}

static void printCsvLine(unsigned long t_ms, int16_t ax, int16_t ay, int16_t az,
                       int16_t gx, int16_t gy, int16_t gz, int16_t temp) {
  Serial.print(t_ms);
  Serial.print(',');
  Serial.print(ax);
  Serial.print(',');
  Serial.print(ay);
  Serial.print(',');
  Serial.print(az);
  Serial.print(',');
  Serial.print(gx);
  Serial.print(',');
  Serial.print(gy);
  Serial.print(',');
  Serial.print(gz);
  Serial.print(',');
  Serial.println(temp);
}

static void processCommand(const String &lineIn) {
  String line = lineIn;
  line.trim();
  if (line.length() == 0)
    return;

  String up = line;
  up.toUpperCase();

  if (up == "START_STREAM") {
    if (!mpuPresent) {
      Serial.println("# START_STREAM отклонён: MPU6050 не найден");
      return;
    }
    streaming = true;
    lastStreamMs = 0;
    Serial.println("# STREAM_START");
    return;
  }

  if (up == "STOP_STREAM") {
    streaming = false;
    Serial.println("# STREAM_STOP");
    return;
  }

  if (up == "WHOAMI") {
    handleWhoAmI();
    return;
  }

  if (up == "RESET") {
    streaming = false;
    Serial.println("# Выполняется RESET MPU6050...");
    mpuPresent = initMpu6050();
    return;
  }

  Serial.print("# Неизвестная команда: ");
  Serial.println(line);
}

void setup() {
  Serial.begin(115200);
  unsigned long t0 = millis();
  while (!Serial && (millis() - t0 < 3000)) {
    delay(10);
  }

  Wire.begin();
  Wire.setClock(400000);

  Serial.println("# mpu6050_stream — готов. Команды: START_STREAM | STOP_STREAM | WHOAMI | RESET");

  mpuPresent = pingMpu6050();
  if (!mpuPresent) {
    Serial.println("# ВНИМАНИЕ: MPU6050 не обнаружен (проверьте A4/A5, питание, AD0→GND).");
    Serial.println("# Доступны команды WHOAMI/RESET для повторной попытки.");
    return;
  }

  if (!initMpu6050())
    mpuPresent = false;
}

void loop() {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\r')
      continue;
    if (c == '\n') {
      processCommand(cmdBuf);
      cmdBuf = "";
    } else {
      if (cmdBuf.length() < 96)
        cmdBuf += c;
    }
  }

  if (!streaming || !mpuPresent)
    return;

  unsigned long now = millis();
  if (lastStreamMs != 0 && (now - lastStreamMs) < STREAM_INTERVAL_MS)
    return;
  lastStreamMs = now;

  uint8_t buf[14];
  if (!readBurst(RA_ACCEL_XOUT_H, buf, 14)) {
    Serial.println("# ERROR: сбой чтения ACCEL..GYRO burst — STOP_STREAM");
    streaming = false;
    return;
  }

  int16_t ax = be16(buf + 0);
  int16_t ay = be16(buf + 2);
  int16_t az = be16(buf + 4);
  int16_t temp = be16(buf + 6);
  int16_t gx = be16(buf + 8);
  int16_t gy = be16(buf + 10);
  int16_t gz = be16(buf + 12);

  printCsvLine(now, ax, ay, az, gx, gy, gz, temp);
}
