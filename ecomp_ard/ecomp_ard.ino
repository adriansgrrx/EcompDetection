#include <AFMotor.h>
#include <Servo.h>

// Servo setup
Servo sorterServo;
const int servoPin = 9;
// Dual Motor Driver (e.g., L298N) Pins
const int ENA = A0;  // Changed to analog pin to avoid shield conflict
const int IN1 = A1;  // Changed to analog pin to avoid shield conflict
const int IN2 = A2;  // Changed to analog pin to avoid shield conflict

const int neutralAngle = 110;

const int motorSpeed = 150;

// L293D Shield Motors on M1-M4
AF_DCMotor motor1(1);
AF_DCMotor motor2(2);
AF_DCMotor motor3(3);
AF_DCMotor motor4(4);
bool motorRunning = false;
unsigned long lastMotorToggleTime = 0;



void setup() {
  Serial.begin(9600);
  
  // Servo initialization
  sorterServo.attach(servoPin);
  sorterServo.write(neutralAngle);
  delay(1000);
  
  // L293D Motors setup
  motor1.setSpeed(255);
  motor1.run(RELEASE);  // M1 will run in loop
  motor2.setSpeed(255);
  motor2.run(RELEASE);
  motor3.setSpeed(255);
  motor3.run(RELEASE);
  motor4.setSpeed(255);
  motor4.run(RELEASE);
  
  // Dual Motor Driver pin setup
  pinMode(ENA, OUTPUT);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  
  // Optional: Start the dual motor idle
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  analogWrite(ENA, 0);
  
  Serial.println("READY");
}

void loop() {
  handleServoCommand();
  //handleMotor1Loop();      // L293D Shield Motor M1 logic
  handleExternalMotor();   // L298N motor logic (1.5s on, 2s off cycle)
}

// --- Servo command handler ---
void handleServoCommand() {
  if (Serial.available() > 0) {
    char command = Serial.read();
    int angle = neutralAngle;
    switch (command) {
      case 'A':
        delay(500);
        angle = 70;
        Serial.println("Moving to BJT bin...");
        break;
      case 'B':
        delay(500);
        angle = 30;
        Serial.println("Moving to LED bin...");
        break;
      case 'C':
        delay(500);
        angle = 180;
        Serial.println("Moving to Capacitor bin...");
        break;
      case 'D':
        delay(500);
        angle = 0;
        Serial.println("Moving to Defective bin...");
        break;
      case 'E':
        delay(500);
        angle = 145;
        Serial.println("Moving to Resistor bin...");
        break;
      case 'U':
        delay(500);
        angle = neutralAngle;
        Serial.println("Moving to Unknown bin...");
        break;
      default:
        Serial.println("Invalid command.");
        flushInput();
        return;
    }
    sorterServo.write(angle);
    delay(1000);
    Serial.println("DONE");
    flushInput();
  }
}

// --- M1 Motor: 1.5s on, 2s off cycle ---
void handleMotor1Loop() {
  unsigned long currentMillis = millis();
  if (motorRunning && currentMillis - lastMotorToggleTime >= 1500) {
    motor1.run(RELEASE);
    motorRunning = false;
    lastMotorToggleTime = currentMillis;
  } else if (!motorRunning && currentMillis - lastMotorToggleTime >= 2000) {
    motor1.run(FORWARD);
    motorRunning = true;
    lastMotorToggleTime = currentMillis;
  }
}

// --- External dual motor driver logic with cycle (1.5s on, 2s off) ---
void handleExternalMotor() {
  unsigned long currentMillis = millis();
  
  if (motorRunning && currentMillis - lastMotorToggleTime >= 750) {
    // Turn motor off
    analogWrite(ENA, 0);
    motorRunning = false;
    lastMotorToggleTime = currentMillis;
  } else if (!motorRunning && currentMillis - lastMotorToggleTime >= 1500) {
    // Turn motor on
    // Set the direction first
    digitalWrite(IN1, HIGH);
    digitalWrite(IN2, LOW);
    // Then set the speed
    analogWrite(ENA, motorSpeed);  // Full speed
    motorRunning = true;
    lastMotorToggleTime = currentMillis;
  }
}

// --- Flush serial input buffer ---
void flushInput() {
  while (Serial.available() > 0) {
    Serial.read();
  }
}