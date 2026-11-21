#pragma once
#include <Arduino.h>

#define WIFI_SSID            "DTEL_2.4G_MATOS"
#define WIFI_PASSWORD        "Ribamar23"

#define IP_ADDRESS(a, b, c, d) IPAddress(a, b, c, d)

#define LOCAL_IP         IP_ADDRESS(192, 168, 1, 11)
#define GATEWAY          IP_ADDRESS(192, 168, 1, 1)
#define SUBNET           IP_ADDRESS(255, 255, 255, 0)
#define PRIMARY_DNS      IP_ADDRESS(192, 168, 1, 1)