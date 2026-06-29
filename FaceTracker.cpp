#include "FaceTracker.h"
#include "SerialComm.h"
#include <iostream>

using namespace cv;
using namespace cv::face;
using namespace std;

FaceTracker::FaceTracker() : isTrained(false), sampleCount(0), lastRecognized(""), registrationRequested(false) {
    recognizer = LBPHFaceRecognizer::create();
}

bool FaceTracker::init(const string& cascadePath) {
    if (!face_cascade.load(cascadePath)) return false;
    try {
        recognizer->read("trained_model.yml");
        isTrained = true;
    } catch (...) {
        cout << "No model found. Register face first." << endl;
    }
    return true;
}

void FaceTracker::startRegistration() {
    if (!isTrained) {
        sampleCount = 0;
        trainingImages.clear();
        trainingLabels.clear();
        registrationRequested = true;
        cout << "Starting registration..." << endl;
    }
}

void FaceTracker::process(Mat& frame, char key, const string& arduinoPort) {
    Mat gray;
    cvtColor(frame, gray, COLOR_BGR2GRAY);
    equalizeHist(gray, gray);

    vector<Rect> faces;
    face_cascade.detectMultiScale(gray, faces, 1.1, 4, 0, Size(100, 100));

    if (key == 'r' || key == 'R') startRegistration();

    for (size_t i = 0; i < faces.size(); i++) {
        Mat faceROI = gray(faces[i]);
        resize(faceROI, faceROI, Size(200, 200));

        string displayText = "Scanning...";
        Scalar color = Scalar(255, 255, 0);

        if (!isTrained && registrationRequested && sampleCount < totalSamplesNeeded) {
            trainingImages.push_back(faceROI.clone());
            trainingLabels.push_back(1);
            sampleCount++;
            
            displayText = "Registering: " + to_string(sampleCount) + "/" + to_string(totalSamplesNeeded);
            color = Scalar(0, 165, 255);

            if (sampleCount == totalSamplesNeeded) {
                recognizer->train(trainingImages, trainingLabels);
                recognizer->save("trained_model.yml");
                isTrained = true;
                registrationRequested = false;
                cout << "Face saved!" << endl;
            }
        } 
        else if (isTrained) {
            int label = -1;
            double confidence = 0.0;
            recognizer->predict(faceROI, label, confidence);

            if (label == 1 && confidence < 70.0) {
                displayText = "User " + to_string(label) + " - Present";
                color = Scalar(0, 255, 0);
                if (lastRecognized != displayText) {
                    sendToArduino(arduinoPort, "1\n");
                    lastRecognized = displayText;
                }
            } else {
                displayText = "Unknown Face";
                color = Scalar(0, 0, 255);
                if (lastRecognized != displayText) {
                    sendToArduino(arduinoPort, "0\n");
                    lastRecognized = displayText;
                }
            }
        }

        rectangle(frame, faces[i], color, 2);
        putText(frame, displayText, Point(faces[i].x, faces[i].y - 10), FONT_HERSHEY_SIMPLEX, 0.8, color, 2);
    }
}