#ifndef SERIAL_COMM_H
#define SERIAL_COMM_H

#include <string>

void sendToArduino(const std::string& portName, const std::string& data);

#endif