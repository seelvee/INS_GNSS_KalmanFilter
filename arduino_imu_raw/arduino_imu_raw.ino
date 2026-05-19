/*
 * Вывод только чистых (сырых) значений акселерометра и гироскопа для ИНС+ГНСС.
 * Обработка (ориентация, перевод в навигационную СК) — на ПК в Python.
 *
 * Формат строки: ax,ay,az,gx,gy,gz (целые LSB с датчика)
 * MPU6050 по I2C. Адрес по умолчанию 0x68.
 */

#include <Wire.h>

const uint8_t MPU6050_ADDR = 0x68;
const uint8_t PWR_MGMT_1   = 0x6B;
const uint8_t ACCEL_XOUT_H = 0x3B;
const uint8_t GYRO_XOUT_H  = 0x43;

void setup() {
  Serial.begin(115200);
  while (!Serial) delay(10);

  Wire.begin();
  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(PWR_MGMT_1);
  Wire.write(0);
  Wire.endTransmission(true);
  delay(100);
}

void loop() {
  int16_t ax, ay, az, gx, gy, gz;

  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(ACCEL_XOUT_H);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU6050_ADDR, (uint8_t)14, (uint8_t)true);

  ax = Wire.read() << 8 | Wire.read();
  ay = Wire.read() << 8 | Wire.read();
  az = Wire.read() << 8 | Wire.read();
  Wire.read(); Wire.read();  // temp
  gx = Wire.read() << 8 | Wire.read();
  gy = Wire.read() << 8 | Wire.read();
  gz = Wire.read() << 8 | Wire.read();

  Serial.print(ax);
  Serial.print(",");
  Serial.print(ay);
  Serial.print(",");
  Serial.print(az);
  Serial.print(",");
  Serial.print(gx);
  Serial.print(",");
  Serial.print(gy);
  Serial.print(",");
  Serial.println(gz);

  delay(10);
}
