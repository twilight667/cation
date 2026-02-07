#include <WiFi.h>
#include <HTTPClient.h>

#define ECG_PIN 34
#define LO_PLUS 2
#define LO_MINUS 18

const char* ssid = "VITC-EVENT";
const char* password = "Eve@07&08#$";
const char* TOKEN = "BBUS-knlzYuuqcUCFhoDIgqf4t6vfIwo6l5";

const char* DEVICE_LABEL = "esp32";   // <-- important
const char* VAR_LABEL = "ecg";        // this will be created automatically

void setup() {
  Serial.begin(115200);
  pinMode(LO_PLUS, INPUT);
  pinMode(LO_MINUS, INPUT);

  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected");
}

void loop() {
  if (digitalRead(LO_PLUS) || digitalRead(LO_MINUS)) {
    delay(50);
    return;
  }

  int ecg = analogRead(ECG_PIN);

  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    String url = "http://industrial.api.ubidots.com/api/v1.6/devices/" + String(DEVICE_LABEL);
    http.begin(url);
    http.addHeader("Content-Type", "application/json");
    http.addHeader("X-Auth-Token", TOKEN);

    // This line CREATES the variable
    String payload = "{\"" + String(VAR_LABEL) + "\":" + String(ecg) + "}";
    http.POST(payload);
    http.end();
  }

  delay(10); // ~100 Hz
}
