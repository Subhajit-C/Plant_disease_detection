# 🌱 Plant Disease Detection using Deep Learning & ML Models

This project is a plant disease classification system that detects diseases in **tomato, potato, and sugarcane** leaves using various **deep learning (CNN)** and **machine learning (SVM, Random Forest)** models. It includes a user-friendly **Flask web interface** for uploading leaf images and viewing disease predictions and detailed information.

---

## 🎬 Video Demo

👉 [Watch Demo](https://drive.google.com/file/d/1Y4MQtlwxtyItxTO0JLqpneJtOhWnz3V0/view?usp=sharing)  
*A quick walkthrough of the features, model predictions, and web interface.*

---

## 🚀 Features

- 🌾 **Supports Multiple Crops**: Tomato, Potato, Sugarcane
- 🧠 **Multiple Models**:
  - CNNs: ResNet50, VGG, DenseNet, Inception, InceptionResNetV2, MobileNetV2
  - ML: SVM, Random Forest
- 📊 **Interactive Web App** with Flask
- 🧾 **Disease Info Display** from JSON files
- 📷 Upload leaf image and get disease name + details
- 👥 Team showcase via `teams.html`

---

## 💡 Novelty

Unlike traditional plant disease detectors that focus on a single crop and use one fixed model, our project offers:

- **Multi-crop selection** (Tomato, Potato, Sugarcane)
- **Model flexibility** – users choose from multiple CNN and ML models
- **Educational disease insights** – includes name, causes, prevention, treatment, and resources

This makes the system more interactive, customizable, and informative compared to standard solutions.

---

## 🌾 Dataset Links

- 🍅 **Tomato Dataset**: [New Plant Diseases Dataset (Augmented)](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
- 🥔 **Potato Dataset**: [PlantVillage Dataset (Potato - Color)](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
- 🍬 **Sugarcane Dataset**: [Custom Sugarcane Leaf Dataset (Google Drive)](https://www.kaggle.com/datasets/nirmalsankalana/sugarcane-leaf-disease-dataset)

---

## 💻 Tech Stack

- **Frontend**: HTML, CSS (via Flask templates)
- **Backend**: Python, Flask
- **Deep Learning**: TensorFlow/Keras
- **Machine Learning**: Scikit-learn
- **Visualization**: Matplotlib, Seaborn (in notebooks)
- **Others**: JSON, PIL, NumPy

---

## ⚙️ Setup Instructions

> 💡 **Note:** This project requires **Python 3.10.0** and **TensorFlow 2.10.0**.  
> Latest versions (e.g., Python 3.13 / TensorFlow 2.16+) may cause compatibility issues.

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/plant_disease_detection.git
cd plant_disease_detection
# Create virtual environment
python -m venv plant_env

# Activate (for Windows)
plant_env\Scripts\activate

# Activate (for Linux/macOS)
source plant_env/bin/activate

# Isntall dependencies
pip install -r requirements.txt

# Run the flask app
python app.py
```

---

## 📦 requirements.txt Summary
This project uses:

- tensorflow==2.10.0 (for deep learning models)
- Flask==3.1.0 (for backend server)
- scikit-learn==1.6.1 (for SVM, Random Forest)
- matplotlib, seaborn, opencv-python, pillow for image handling and analysis

Refer to the full requirements.txt file for exact versions.

---

## 📝 License
This project is for educational purposes.

