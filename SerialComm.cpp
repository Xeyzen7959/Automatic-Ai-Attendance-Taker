#include "SerialComm.h"
#include <unistd.h>
#include <fcntl.h>
#include <termios.h>

void sendToArduino(const std::string& portName, const std::string& data) {
    int fd = open(portName.c_str(), O_WRONLY | O_NOCTTY);
    if (fd != -1) {
        struct termios tty;
        tcgetattr(fd, &tty);
        cfsetospeed(&tty, B9600);
        tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8;
        tty.c_cflag &= ~PARENB;
        tty.c_cflag &= ~CSTOPB;
        tcsetattr(fd, TCSANOW, &tty);
        
        write(fd, data.c_str(), data.length());
        close(fd);
    }
}