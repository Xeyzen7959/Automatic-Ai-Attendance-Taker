#include "FaceTracker.h"
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Error: Cannot open webcam." << endl;
        return -1;
    }

    FaceTracker tracker;
    if (!tracker.init("haarcascade_frontalface_default.xml")) {
        cerr << "Error: Cannot load XML cascade file." << endl;
        return -1;
    }

    string arduinoPort = "/dev/cu.usbmodem101"; 
    Mat frame;

    cout << "Press 'R' in the graphics window to register a new student profile. Press 'ESC' to close application." << endl;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        char key = (char)waitKey(10);
        if (key == 27) break;

        tracker.process(frame, key, arduinoPort);
        imshow("Facial Recognition Attendance Simulation", frame);
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
