# 🏷️ Facial Recognition Attendance Tracker

<p align="center">
  <img src="https://img.shields.io/badge/C%2B%2B-11-blue?style=for-the-badge&logo=c%2B%2B" alt="C++">
  <img src="https://img.shields.io/badge/OpenCV-4.x-green?style=for-the-badge&logo=opencv" alt="OpenCV">
  <img src="https://img.shields.io/badge/Arduino-Uno-00979D?style=for-the-badge&logo=arduino" alt="Arduino">
  <img src="https://img.shields.io/badge/Platform-Windows%20%7C%20macOS-lightgrey?style=for-the-badge" alt="Platform">
</p>

---

## 📝 Product Description
An AI-powered automated attendance system that replaces manual logbooks with real-time biometric verification. Running a high-speed **Face Detection and Recognition pipeline** (Haar Cascades + LBPH), the system identifies registered individuals via a webcam and instantly transmits authorization payloads to an Arduino hardware layer to manage localized access or displays.

---

## 👥 Team & Technical Roles

| Name | Professional Role | Core Engineering Responsibilities |
| :--- | :--- | :--- |
| **Joshua Nathanael Tampubolon** | **Core AI & Systems Architect** | • Engineered the backend computer vision pipeline utilizing OpenCV.<br>• Integrated the Local Binary Patterns Histograms (LBPH) mathematical model.<br>• Developed the primary system engine and OS-specific compilation drivers. |
| **Aditya Saputra Pambudi** | **Lead UI/UX & Graphics Engineer** | • Architected the runtime visual overlay and interactive state machine.<br>• Programmed the dynamic bounding box alert matrices (Green/Yellow/Red status states).<br>• Managed frame asset scaling, real-time telemetry rendering, and interface layouts. |
| **M Ikhsan Ar Rahman** | **Embedded Systems & Firmware Engineer** | • Authored the Arduino micro-controller firmware to parse inbound serial data packets.<br>• Managed memory addresses and data pipelines for the I2C 16x2 LCD layout engine.<br>• Engineered hardware wire topologies, debouncing logic, and electrical schematics. |
| **Fadil Hibrian Pratama** | **QA Engineer & Optimization Analyst** | • Executed automated benchmarking suites to evaluate frame-per-second latency drop.<br>• Fine-tuned the face-confidence threshold hyperparameters to eliminate false positives.<br>• Conducted security penetration testing (e.g., photo-spoofing tests) and edge-case validation. |

---

## 📂 Project Structure

```text
AttendanceProject/
├── 📄 SerialComm.h        # Serial interface definitions
├── 📄 SerialComm.cpp      # Win32/POSIX OS-specific serial drivers
├── 📄 FaceTracker.h       # LBPH Tracking engine definitions
├── 📄 FaceTracker.cpp     # Detection, registration, and training logic
├── 📄 main.cpp            # Application lifecycle entryway
└── ⚙️ haarcascade_frontalface_default.xml
