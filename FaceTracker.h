
#ifndef FACE_TRACKER_H
#define FACE_TRACKER_H

#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <vector>
#include <string>

class FaceTracker {
public:
    FaceTracker();
    bool init(const std::string& cascadePath);
    void startRegistration();
    void process(cv::Mat& frame, char key, const std::string& arduinoPort);

private:
    cv::CascadeClassifier face_cascade;
    cv::Ptr<cv::face::LBPHFaceRecognizer> recognizer;
    bool isTrained;
    std::vector<cv::Mat> trainingImages;
    std::vector<int> trainingLabels;
    int sampleCount;
    const int totalSamplesNeeded = 30;
    std::string lastRecognized;
    bool registrationRequested;
};

#endif