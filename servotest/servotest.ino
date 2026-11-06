#include <Wire.h>
#include <Servo.h>

const uint8_t MPU_ADDR = 0x68;
const int SERVO_PIN = 9;

Servo linearServo;

float hpAccelX = 0.0;
float prevAccX = 0.0;

// Filter & Control Parameters
float leakFactor = 0.85; // High-pass coefficient (bleeds off static gravity/tilt & slow writing)
float Kp = 40.0;         // Gain factor for linear acceleration compensation

// Safe Stroke Limits (Adjust for your actuator lead screw)
const int MIN_STROKE = 30;
const int MAX_STROKE = 150;

void setup() {
  Serial.begin(115200);
  Wire.begin();
  Wire.setWireTimeout(3000, true); // Prevent I2C lockup

  // Wake up MPU6050
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B);
  Wire.write(0);
  Wire.endTransmission(true);

  linearServo.attach(SERVO_PIN);
  linearServo.write(90); // Center baseline position (mid-stroke)

  // Initialize baseline reading
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU_ADDR, 2, true);
  int16_t rawAccX = Wire.read() << 8 | Wire.read();
  prevAccX = rawAccX / 16384.0;
}

void loop() {
  // Read Accelerometer X-axis (Register 0x3B)
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU_ADDR, 2, true);

  int16_t rawAccX = Wire.read() << 8 | Wire.read();
  
  // Convert to g-force (+/- 2g range scale factor = 16384.0 LSB/g)
  float currentAccX = -rawAccX / 16384.0; 

  // High-pass filter: passes fast linear twitches, blocks gravity & slow hand movement
  hpAccelX = leakFactor * (hpAccelX + currentAccX - prevAccX);
  prevAccX = currentAccX;

  // Noise deadband (ignores subtle resting vibration below 0.02g)
  float activeAccel = hpAccelX;
  if (abs(activeAccel) < 0.02) {
    activeAccel = 0.0;
  }

  // Compute corrective linear position
  int strokePosition = 90 - (int)(activeAccel * Kp);

  // Constrain to prevent mechanical lead screw jamming
  strokePosition = constrain(strokePosition, MIN_STROKE, MAX_STROKE);

  linearServo.write(strokePosition);

  // Serial Monitor / Plotter output
  Serial.print("Linear_Tremor_X:");
  Serial.print(activeAccel * 100); // Scaled for plotter visibility
  Serial.print(",");
  Serial.print("Servo_Pos:");
  Serial.println(strokePosition);

  delay(10); // ~100Hz refresh rate
}