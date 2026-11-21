#include <Arduino.h>
#include <WebServer.h>
#include <WiFi.h>

#include "esp32-hal-gpio.h"
#include "gpio_pins.h"
#include "http_parser.h"
#include "secrets.h"

#define TRACE(...)    Serial.printf(__VA_ARGS__)
#define RELAY_ON      LOW
#define RELAY_OFF     HIGH

void activateRelay();
void deactivateRelay();
void connectWifi();

WebServer server(80);

void activateRelay(void) { digitalWrite(RELAY_PORT, RELAY_ON); }

void deactivateRelay(void) { digitalWrite(RELAY_PORT, RELAY_OFF); }

void connectWifi() {
  WiFi.mode(WIFI_STA);
  WiFi.config(LOCAL_IP, GATEWAY, SUBNET, PRIMARY_DNS);

  if (strlen(WIFI_SSID) == 0) {
    WiFi.begin();
  } else {
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  }

  TRACE("Connecting to WiFi...\n");


  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    TRACE(".");
  }

  TRACE("connected.\n");
}

void setup(void) {
  delay(3000);

  Serial.begin(115200);
  Serial.setDebugOutput(false);

  pinMode(RELAY_PORT, OUTPUT);
  digitalWrite(RELAY_PORT, RELAY_OFF);

  connectWifi();

  TRACE("Register service handlers...\n");

  server.on("/api/activate", HTTP_GET, []() {
    activateRelay();

    TRACE("Relé ativado.\n");

    server.send(200, "text/plain", "RELAY_ACTIVATED");
  });

  server.on("/api/deactivate", HTTP_GET, []() {
    deactivateRelay();

    TRACE("Relé desativado.\n");

    server.send(200, "text/plain", "RELAY_DEACTIVATED");
  });

  server.begin();

  TRACE("open <http://%s> or <http://%s>\n", WiFi.getHostname(),
        WiFi.localIP().toString().c_str());
}

void loop(void) { server.handleClient(); }
