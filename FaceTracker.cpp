#include "FaceTracker.h"
#include "SerialComm.h"
#include <iostream>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace cv::face;
using namespace std;

FaceTracker::FaceTracker() : isTrained(false), sampleCount(0), lastRecognized(""), registrationRequested(false), currentRegisteringId(1) {
    recognizer = LBPHFaceRecognizer::create();
}

void FaceTracker::loadDatabase() {
    studentDatabase.clear();
    ifstream file("students.csv");
    if (!file.is_open()) return;

    string line;
    int maxId = 0;
    while (getline(file, line)) {
        stringstream ss(line);
        string idStr, name, nim;
        if (getline(ss, idStr, ',') && getline(ss, name, ',') && getline(ss, nim, ',')) {
            int id = stoi(idStr);
            studentDatabase[id] = {name, nim};
            if (id > maxId) maxId = id;
        }
    }
    file.close();
    currentRegisteringId = maxId + 1; // Auto-increment ID for the next student
}

void FaceTracker::saveToDatabase(int id, const string& name, const string& nim) {
    ofstream file("students.csv", ios::app);
    if (file.is_open()) {
        file << id << "," << name << "," << nim << "\n";
        file.close();
    }
    studentDatabase[id] = {name, nim};
}

bool FaceTracker::init(const string& cascadePath) {
    if (!face_cascade.load(cascadePath)) return false;
    
    loadDatabase(); // Load the name/NIM text mappings first
    
    try {
        recognizer->read("trained_model.yml");
        isTrained = true;
    } catch (...) {
        cout << "No model found. Register face first." << endl;
    }
    return true;
}

void FaceTracker::startRegistration() {
    string name, nim;
    cout << "\n=== NEW STUDENT REGISTRATION ===" << endl;
    cout << "Enter Student Name: ";
    getline(cin, name); // Removed cin.ignore() which was skipping the first letter
    cout << "Enter Student NIM: ";
    getline(cin, nim);

    sampleCount = 0;
    trainingImages.clear();
    trainingLabels.clear();
    registrationRequested = true;
    
    saveToDatabase(currentRegisteringId, name, nim);
    cout << "Data saved to database. Look at the camera to capture face profile..." << endl;
}

void FaceTracker::process(Mat& frame, char key, const string& arduinoPort) {
    Mat gray;
    cvtColor(frame, gray, COLOR_BGR2GRAY);
    equalizeHist(gray, gray);

    vector<Rect> faces;
    face_cascade.detectMultiScale(gray, faces, 1.1, 4, 0, Size(100, 100));

    if (key == 'r' || key == 'R') {
        startRegistration();
        return; 
    }

    for (size_t i = 0; i < faces.size(); i++) {
        Mat faceROI = gray(faces[i]);
        resize(faceROI, faceROI, Size(200, 200));

        string displayText = "Scanning...";
        Scalar color = Scalar(255, 255, 0);

        if (registrationRequested && sampleCount < totalSamplesNeeded) {
            // Train with existing images if model exists, plus the new ones
            trainingImages.push_back(faceROI.clone());
            trainingLabels.push_back(currentRegisteringId);
            sampleCount++;
            
            displayText = "Registering: " + to_string(sampleCount) + "/" + to_string(totalSamplesNeeded);
            color = Scalar(0, 165, 255);

            if (sampleCount == totalSamplesNeeded) {
                // Read old model data if it exists to append new face structural weights
                try {
                    recognizer->read("trained_model.yml");
                    recognizer->update(trainingImages, trainingLabels);
                } catch(...) {
                    recognizer->train(trainingImages, trainingLabels);
                }
                
                recognizer->save("trained_model.yml");
                isTrained = true;
                registrationRequested = false;
                cout << "Face profile trained and locked successfully for ID: " << currentRegisteringId << "!\n" << endl;
                currentRegisteringId++; 
            }
        } 
        else if (isTrained && !registrationRequested) {
            int label = -1;
            double confidence = 0.0;
            recognizer->predict(faceROI, label, confidence);

            // Check if label exists in our map and passes confidence verification threshold
            if (studentDatabase.find(label) != studentDatabase.end() && confidence < 70.0) {
                Student s = studentDatabase[label];
                displayText = s.name + " (" + s.nim + ") - Present";
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
        putText(frame, displayText, Point(faces[i].x, faces[i].y - 10), FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
    }
}
