/*
 * ESP32 Servo Controller for Tomato Harvesting Robot
 * Supports both Serial and WiFi control
 * 
 * Hardware Requirements:
 * - ESP32 Development Board
 * - 6x Servo Motors (connected to pins defined below)
 * - External 5V power supply for servos
 * 
 * Libraries Required:
 * - ESP32Servo
 * - ArduinoJson
 * - WiFi (built-in)
 * - WebServer (built-in)
 */

#include <ESP32Servo.h>
#include <ArduinoJson.h>
#include <WiFi.h>
#include <WebServer.h>

// WiFi Configuration
const char* ssid = "YourWiFiSSID";      // Change this
const char* password = "YourWiFiPassword"; // Change this

// Servo Configuration
#define NUM_SERVOS 4
const int servoPins[NUM_SERVOS] = {13, 12, 14, 27}; // GPIO pins for servos
Servo servos[NUM_SERVOS];
int servoPositions[NUM_SERVOS] = {90, 90, 90, 90}; // Current positions
int servoMin = 500;   // Minimum pulse width in microseconds
int servoMax = 2400;  // Maximum pulse width in microseconds

// Servo names for reference
const char* servoNames[NUM_SERVOS] = {
  "Base", "Shoulder", "Elbow", "Gripper"
};

// Movement parameters
int moveSpeed = 50;  // Default movement speed (delay between steps in ms)
bool emergencyStop = false;

// Web Server
WebServer server(80);

// JSON Document for parsing commands
StaticJsonDocument<512> jsonDoc;

// LED for status indication
#define STATUS_LED 2  // Built-in LED on most ESP32 boards

void setup() {
  Serial.begin(115200);
  delay(1000);
  
  // Initialize status LED
  pinMode(STATUS_LED, OUTPUT);
  digitalWrite(STATUS_LED, LOW);
  
  // Initialize servos
  initializeServos();
  
  // Initialize WiFi (optional - comment out if using only Serial)
  initializeWiFi();
  
  // Initialize web server endpoints
  setupWebServer();
  
  Serial.println("{\"status\":\"ready\",\"message\":\"ESP32 Servo Controller Ready\"}");
  blinkLED(3, 200); // Blink 3 times to indicate ready
}

void loop() {
  // Handle Serial commands
  if (Serial.available()) {
    handleSerialCommand();
  }
  
  // Handle WiFi clients
  server.handleClient();
  
  // Check emergency stop
  if (emergencyStop) {
    // Don't move servos during emergency stop
    delay(10);
  }
}

void initializeServos() {
  // Allow allocation of all timers
  ESP32PWM::allocateTimer(0);
  ESP32PWM::allocateTimer(1);
  ESP32PWM::allocateTimer(2);
  ESP32PWM::allocateTimer(3);
  
  // Attach servos to pins
  for (int i = 0; i < NUM_SERVOS; i++) {
    servos[i].setPeriodHertz(50);  // Standard 50hz servo
    servos[i].attach(servoPins[i], servoMin, servoMax);
    servos[i].write(servoPositions[i]);
    delay(100);
  }
  
  Serial.println("{\"status\":\"servos_initialized\"}");
}

void initializeWiFi() {
  Serial.print("Connecting to WiFi");
  WiFi.begin(ssid, password);
  
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    attempts++;
    digitalWrite(STATUS_LED, !digitalRead(STATUS_LED)); // Blink while connecting
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println();
    Serial.print("{\"status\":\"wifi_connected\",\"ip\":\"");
    Serial.print(WiFi.localIP());
    Serial.println("\"}");
    digitalWrite(STATUS_LED, HIGH); // Solid on when connected
  } else {
    Serial.println();
    Serial.println("{\"status\":\"wifi_failed\"}");
    digitalWrite(STATUS_LED, LOW);
  }
}

void setupWebServer() {
  // Status endpoint
  server.on("/status", HTTP_GET, []() {
    StaticJsonDocument<256> response;
    response["status"] = "online";
    response["emergency_stop"] = emergencyStop;
    
    JsonArray positions = response.createNestedArray("positions");
    for (int i = 0; i < NUM_SERVOS; i++) {
      positions.add(servoPositions[i]);
    }
    
    String output;
    serializeJson(response, output);
    server.send(200, "application/json", output);
  });
  
  // Command endpoint
  server.on("/command", HTTP_POST, []() {
    if (server.hasArg("plain")) {
      String body = server.arg("plain");
      DeserializationError error = deserializeJson(jsonDoc, body);
      
      if (!error) {
        processCommand(jsonDoc);
        server.send(200, "application/json", "{\"status\":\"success\"}");
      } else {
        server.send(400, "application/json", "{\"status\":\"error\",\"message\":\"Invalid JSON\"}");
      }
    } else {
      server.send(400, "application/json", "{\"status\":\"error\",\"message\":\"No data\"}");
    }
  });
  
  // Start server
  server.begin();
}

void handleSerialCommand() {
  String command = Serial.readStringUntil('\n');
  command.trim();
  
  if (command.length() > 0) {
    DeserializationError error = deserializeJson(jsonDoc, command);
    
    if (!error) {
      processCommand(jsonDoc);
    } else {
      Serial.println("{\"status\":\"error\",\"message\":\"Invalid JSON\"}");
    }
  }
}

void processCommand(JsonDocument& doc) {
  String cmd = doc["cmd"].as<String>();
  
  if (cmd == "HELLO" || cmd == "PING") {
    Serial.println("{\"status\":\"ready\",\"type\":\"ESP32_SERVO_CONTROLLER\"}");
    
  } else if (cmd == "SERVO_MOVE") {
    int servo = doc["servo"].as<int>();
    int angle = doc["angle"].as<int>();
    int speed = doc["speed"] | moveSpeed;
    
    if (servo >= 0 && servo < NUM_SERVOS && angle >= 0 && angle <= 180) {
      moveServo(servo, angle, speed);
      sendResponse(true, "Servo moved");
    } else {
      sendResponse(false, "Invalid parameters");
    }
    
  } else if (cmd == "SERVO_MOVE_ALL") {
    JsonArray angles = doc["angles"].as<JsonArray>();
    int speed = doc["speed"] | moveSpeed;
    
    if (angles.size() == NUM_SERVOS) {
      int targetAngles[NUM_SERVOS];
      for (int i = 0; i < NUM_SERVOS; i++) {
        targetAngles[i] = angles[i].as<int>();
      }
      moveAllServos(targetAngles, speed);
      sendResponse(true, "All servos moved");
    } else {
      sendResponse(false, "Invalid angles array");
    }
    
  } else if (cmd == "GRIPPER") {
    String action = doc["action"].as<String>();
    handleGripper(action);
    
  } else if (cmd == "HOME") {
    homePosition();
    sendResponse(true, "Moved to home position");
    
  } else if (cmd == "EMERGENCY_STOP") {
    emergencyStopAll();
    sendResponse(true, "Emergency stop activated");
    
  } else if (cmd == "RELEASE_STOP") {
    emergencyStop = false;
    sendResponse(true, "Emergency stop released");
    
  } else if (cmd == "GET_POSITIONS") {
    sendPositions();
    
  } else if (cmd == "SMART_PICK") {
    String type = doc["type"].as<String>();
    int force = doc["force"] | 60;
    String speed = doc["speed"] | "normal";
    smartPick(type, force, speed);
    
  } else if (cmd == "SMART_DISCARD") {
    String type = doc["type"].as<String>();
    smartDiscard(type);
    
  } else {
    sendResponse(false, "Unknown command");
  }
}

void moveServo(int servoIndex, int targetAngle, int speed) {
  if (emergencyStop) return;
  
  int currentAngle = servoPositions[servoIndex];
  int step = (targetAngle > currentAngle) ? 1 : -1;
  
  while (currentAngle != targetAngle && !emergencyStop) {
    currentAngle += step;
    servos[servoIndex].write(currentAngle);
    servoPositions[servoIndex] = currentAngle;
    delay(speed);
  }
}

void moveAllServos(int targetAngles[], int speed) {
  if (emergencyStop) return;
  
  bool allReached = false;
  
  while (!allReached && !emergencyStop) {
    allReached = true;
    
    for (int i = 0; i < NUM_SERVOS; i++) {
      if (servoPositions[i] != targetAngles[i]) {
        allReached = false;
        
        int step = (targetAngles[i] > servoPositions[i]) ? 1 : -1;
        servoPositions[i] += step;
        servos[i].write(servoPositions[i]);
      }
    }
    
    delay(speed);
  }
}

void handleGripper(String action) {
  int gripperIndex = 3; // Gripper is the last servo (4th servo)
  
  if (action == "open") {
    moveServo(gripperIndex, 120, 30);
  } else if (action == "close") {
    moveServo(gripperIndex, 30, 30);
  } else if (action == "grip") {
    // Gentle grip with feedback (simplified)
    moveServo(gripperIndex, 60, 50);
  }
}

void homePosition() {
  int homeAngles[NUM_SERVOS] = {90, 90, 90, 90};
  moveAllServos(homeAngles, 30);
}

void emergencyStopAll() {
  emergencyStop = true;
  // Optionally disable servo signals
  for (int i = 0; i < NUM_SERVOS; i++) {
    servos[i].detach();
  }
  digitalWrite(STATUS_LED, LOW);
}

void smartPick(String type, int force, String speed) {
  // Implement smart picking sequence based on tomato type
  int pickSpeed = 30;
  
  if (speed == "slow") pickSpeed = 50;
  else if (speed == "very_slow") pickSpeed = 70;
  else if (speed == "fast") pickSpeed = 20;
  
  // Example picking sequence for 4 servos
  // 1. Move to approach position
  int approachPos[NUM_SERVOS] = {90, 45, 135, 120};
  moveAllServos(approachPos, pickSpeed);
  delay(500);
  
  // 2. Move to pick position
  int pickPos[NUM_SERVOS] = {90, 30, 150, 120};
  moveAllServos(pickPos, pickSpeed);
  delay(500);
  
  // 3. Close gripper with appropriate force
  int gripAngle = map(force, 0, 100, 30, 90);
  moveServo(3, gripAngle, pickSpeed);
  delay(500);
  
  // 4. Lift
  int liftPos[NUM_SERVOS] = {90, 60, 120, gripAngle};
  moveAllServos(liftPos, pickSpeed);
  delay(500);
  
  // 5. Move to basket
  int basketPos[NUM_SERVOS] = {0, 90, 90, gripAngle};
  moveAllServos(basketPos, pickSpeed);
  delay(500);
  
  // 6. Release
  moveServo(3, 120, pickSpeed);
  delay(500);
  
  // 7. Return to home
  homePosition();
  
  sendResponse(true, "Pick sequence completed");
}

void smartDiscard(String type) {
  // Implement discard sequence for rotten tomatoes
  // Similar to smartPick but moves to discard bin instead
  
  // Example sequence for 4 servos
  int discardSpeed = 50; // Gentle movement
  
  // Move to discard position
  int discardPos[NUM_SERVOS] = {180, 90, 90, 120};
  moveAllServos(discardPos, discardSpeed);
  delay(500);
  
  // Release
  moveServo(3, 120, discardSpeed);
  delay(500);
  
  // Return to home
  homePosition();
  
  sendResponse(true, "Discard sequence completed");
}

void sendPositions() {
  StaticJsonDocument<256> response;
  response["status"] = "success";
  
  JsonArray positions = response.createNestedArray("positions");
  for (int i = 0; i < NUM_SERVOS; i++) {
    JsonObject servo = positions.createNestedObject();
    servo["id"] = i;
    servo["name"] = servoNames[i];
    servo["angle"] = servoPositions[i];
  }
  
  String output;
  serializeJson(response, output);
  Serial.println(output);
}

void sendResponse(bool success, String message) {
  StaticJsonDocument<128> response;
  response["status"] = success ? "success" : "error";
  response["message"] = message;
  
  String output;
  serializeJson(response, output);
  Serial.println(output);
}

void blinkLED(int times, int delayMs) {
  for (int i = 0; i < times; i++) {
    digitalWrite(STATUS_LED, HIGH);
    delay(delayMs);
    digitalWrite(STATUS_LED, LOW);
    delay(delayMs);
  }
}
