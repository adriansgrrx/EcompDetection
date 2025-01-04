#include <Servo.h>

Servo myServo;

void setup() {
  Serial.begin(9600);
  myServo.attach(9);
  myServo.write(90); // Neutral position
  Serial.println("Servo Control Ready! Waiting for commands...");
}

void loop() {
  if (Serial.available() > 0) {
    char command = Serial.read();

    if (command == 'A') { 
      myServo.write(180);  // Bin 1
      Serial.println("BJT detected! Servo moved to Bin 1 (180 degrees).");
    } else if (command == 'B') { 
      myServo.write(150);  // Bin 2
      Serial.println("LED detected! Servo moved to Bin 2 (150 degrees).");
    } else if (command == 'C') { 
      myServo.write(120);  // Bin 3
      Serial.println("Capacitor detected! Servo moved to Bin 3 (120 degrees).");
    } else if (command == 'D') { 
      myServo.write(60);  // Bin 4
      Serial.println("Defective component detected! Servo moved to Bin 4 (90 degrees).");
    } else if (command == 'E') { 
      myServo.write(90);  // Bin 5
      Serial.println("Resistor detected! Servo moved to Bin 5 (60 degrees).");
    } else if (command == 'F') { 
      myServo.write(30);  // Bin 6
      Serial.println("Unknown component detected! Servo moved to Bin 6 (30 degrees).");
    } else {
      Serial.println("Invalid command received.");
    }

    delay(1000);  // Wait for the servo to finish moving
    Serial.println("DONE");  // Notify Python that the component has been dispensed
  }
}
