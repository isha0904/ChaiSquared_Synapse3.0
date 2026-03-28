# ChaiSquared_Synapse3.0

# ♻️ Green Vision — Smart Bin Assistant

> AI-powered Computer Vision system for intelligent waste segregation and sustainable disposal guidance

---

## Theme: Sustainability & Waste Management

Global recycling systems suffer from a major issue called **“wish-cycling”** — where users incorrectly dispose of waste due to lack of awareness about resin codes (1–7) and disposal symbols.

This leads to:

* Contaminated recycling streams
* Increased landfill waste
* Reduced recycling efficiency

---

## Problem Statement

Build a Computer Vision-based **Smart Bin Assistant** that:

* Uses a camera feed
* Identifies waste items in real-time
* Provides clear disposal instructions

---

## Solution Overview

Eco-Label Vision is an AI-driven system that:

1. Detects waste objects using live camera input
2. Classifies them into material categories
3. Provides disposal instructions instantly
4. Tracks environmental impact
5. Suggests nearby recycling centers

---

## System Architecture
<img width="1266" height="743" alt="image" src="https://github.com/user-attachments/assets/e21a43b2-8c23-4dc5-acbb-d0952169e06d" />

---

## Tech Stack

### Computer Vision

* YOLO (fine-tuned) → Object Detection
* MobileNet (fine-tuned) → Material Classification
<img width="1003" height="211" alt="image" src="https://github.com/user-attachments/assets/3fb36bec-fd7e-45fb-ac66-8fdec4c7752a" />

### OCR

* EasyOCR → Detect resin codes (1–7)

### APIs

* Google Maps API → Nearby recycling centers

### Frontend

* Streamlit → UI & real-time interaction

---

## Features

#### Webcam Integration

* Real-time camera input via browser

#### Object Identification

* Detects:

  * Plastic
  * Paper
  * Metal
  * Glass

---


#### Resin Code Recognition

* Detects numbers (1–7) on plastic using OCR
* Improves recycling accuracy

#### Tracks:

  * Items sorted
  * Carbon saved

* Awards badges for progress

#### Recycler Recommendations

* Suggests nearby recycling facilities

---

## Dataset used for fine tuning:  [Waste Classification Dataset](https://www.kaggle.com/datasets/phenomsg/waste-classification/data?)

A dataset of 30,000+ images categorized for Computer Vision models.

---

## Gamification

| Action     | Reward          |
| ---------- | --------------- |
| First item | 🌱 Badge        |
| 10 items   | 🌿 Eco Aware    |
| 50 items   | 🏅 Carbon Saver |

---

## Project Structure

```
eco-label-vision/
│── app.py
│── mobilenetv2.pt
│── YoloModel.pt
│── requirements.txt
│── README.md
│── demo_video.mp4
│── flowchart.png
```

---

## Demo

[Demo Video Link](https://drive.google.com/file/d/1_myW0q5dWfbmDZGLt-VvKLAHKC0l4scq/view?usp=sharing)
---

## Impact

* Reduces recycling mistakes
* Improves sustainability awareness
* Encourages responsible disposal

---

## Future Improvements

* Advanced detection models
* Mobile app integration
* Cloud deployment

---

## Team Chai Squared

- Isha Baviskar
- Neha Chatterjee
- Chaitrali Bhosale
---


"We don’t just classify waste — we guide decisions." ♻️

