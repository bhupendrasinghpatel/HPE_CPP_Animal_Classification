
# HPE CPP Project: Camera Trap Animal Classification using YOLOv8
**By Monish P, D V Vedith Varma, Mahika D, Aastha Priya, Bhupendra Singh**  
🎯 Live Demo: [Click Here](https://hpecppanimalclassification-5s9sy6rkwbwt43tujlx5p2.streamlit.app/) <br>
🎯 Powerpoint Presentation: [Click Here](https://docs.google.com/presentation/d/1212ER9rKRJuO9cSGTBcUy-vfRsgjCFmexNIrkKmXihA/edit?slide=id.p#slide=id.p)
🎯 Project Document: [Click Here](https://docs.google.com/document/d/1sqGlpHL5BWua5tqzlVCZgNkedA5cSsnlBauIx_40JWs/edit?usp=sharing)

---

## 📌 Project Overview

This repository presents a complete pipeline for **automated detection and classification of animals** in camera trap images using **YOLOv8**. The primary objective of the project is to achieve higher mAP score for **small animal species**. The system is deployed via a web interface, making it accessible to conservationists, ecologists, and researchers for real-time wildlife monitoring and species analysis.

---

## 🧠 Motivation

Wildlife monitoring through camera traps generates **millions of unlabelled images**, making manual analysis inefficient and error-prone. This project uses deep learning to:
- Classify small and elusive animals accurately
- Handle class imbalance and occlusion
- Work in diverse lighting and natural environments

---

## 🧪 Core Features

✅ Real-time animal species classification via a web app  
✅ YOLOv8l fine-tuned on wildlife datasets  
✅ High accuracy even for small or partially visible species  
✅ Scalable and field-deployable system  
✅ Clean UI for easy image upload and predictions  

---

## 🐾 Dataset Details

- 📦 Source: [LILA BC - Snapshot Serengeti](https://lila.science/datasets/snapshot-serengeti), [iWildCam 2022](https://lila.science/datasets/iwildcam-2022/)
- 🌍 Diversity: 40+ species including Agouti, Bushbuck, Ocelot, Hare, Paca, etc.
- 🧾 Labels: Species name + bounding box coordinates
- 🔧 Format: Converted to YOLO-compatible format

---

## ⚙️ Tech Stack & Dependencies

### 💼 Frameworks and Libraries
- Python 3.10+
- Ultralytics YOLOv8
- Streamlit
- OpenCV
- NumPy
- Pillow
- PyYAML
- Torch & torchvision
- Matplotlib

### 📄 requirements.txt
```
ultralytics==8.0.206
streamlit==1.28.2
opencv-python-headless
Pillow
numpy
PyYAML
torch>=2.0
matplotlib
```

---

## 🛠️ How to Run Locally

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/HPE_CPP_ANIMAL_CLASSIFICATION.git
cd HPE_CPP_ANIMAL_CLASSIFICATION
```

### 2️⃣ Set Up Python Environment
```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3️⃣ Run the App
```bash
streamlit run app.py
```

📂 Make sure the `weights/` folder exists with the trained YOLOv8 model file (`best.pt` or similar).

---

## 🚀 Model Training Details

- 📐 Input size: `960x960`
- 🧠 Model: `YOLOv8l`
- 🗂️ Dataset: Custom YOLO-formatted wildlife data
- 🧪 Epochs: `50`, Batch size: `4`
- 💻 Command used:
```bash
yolo task=detect mode=train model=yolov8l.pt data=custom.yaml imgsz=960 epochs=50 batch=4
```

---

## 📊 Performance (YOLOv8l)

| Metric        | Score |
|---------------|-------|
| Precision     | 0.93  |
| Recall        | 0.82  |
| mAP@0.5       | 0.90  |
| mAP@0.5:0.95  | 0.79  |

---

## 📈 YOLOv8 Model Comparison

| Feature                     | YOLOv8n | YOLOv8m | **YOLOv8l** |
|----------------------------|---------|---------|--------------|
| Model Size                 | 3.2 MB  | 25.4 MB | 52.4 MB     |
| Inference Speed (FPS)      | 300+    | 150–200 | 80–120      |
| Detection Accuracy         | 52–56%  | 61–65%  | **66–70%**  |
| Small Animal Detection     | Poor    | Good    | **Excellent** |
| Class Imbalance Handling   | Weak    | Moderate| **Strong**   |

---

## 🎯 Future Improvements

- Build a desktop and mobile interface for offline use  
- Integrate temporal filtering to reduce false positives  
- Improve UI with map overlays and location tagging  
- Add support for multiple image uploads and batch predictions  

---

## 📚 References

- [YOLOv8 - Ultralytics](https://github.com/ultralytics/ultralytics)  
- [Snapshot Serengeti Dataset](https://lila.science/datasets/snapshot-serengeti)  
- [iWildCam Dataset](https://lila.science/datasets/iwildcam-2022)  

---

## 🤝 Contributing

Want to contribute? Fork the repository and create a pull request. Ideas, suggestions, and improvements are always welcome.

---

## 🛡 License

This project is open-sourced under the [MIT License](LICENSE).

---

📬 For questions or feedback, raise an issue or contact the project maintainers.
