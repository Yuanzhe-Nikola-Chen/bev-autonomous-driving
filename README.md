# 🚗 BEV Autonomous Driving Algorithms

A research-oriented implementation of Bird’s Eye View (BEV) perception and data processing pipeline for autonomous driving systems, with a focus on **control, planning, and energy-aware optimization**.

---

## 📌 Overview

This project aims to build a **clean, modular, and extensible BEV-based pipeline** for autonomous driving research, integrating:

* Multi-sensor data preprocessing (camera-centric)
* BEV transformation logic
* Data augmentation strategies for robust perception
* Foundations for downstream tasks such as:

  * Motion planning
  * Control
  * Energy-aware decision-making

---

## 🔬 Research Motivation

Modern autonomous driving systems rely heavily on **BEV representations** for:

* Spatial reasoning
* Sensor fusion
* Robust planning under uncertainty

This repository is designed as a **research playground** for:

* Control-oriented autonomy
* V2X-integrated systems
* Energy-aware fleet optimization

---

## 🧠 Key Features

* ✅ Clean dataset abstraction for NuScenes-style data
* ✅ Scene filtering and temporal ordering
* ✅ Training vs validation pipeline separation
* ✅ Advanced data augmentation:

  * Random resize
  * Spatial cropping with geometric bias
  * Horizontal flipping
  * Rotation perturbations
* ✅ Reproducible preprocessing pipeline

---

## 📂 Project Structure

```
bev-autonomous-driving/
│
├── bev_dataset.py          (Current Progress)
├── bev_geometry.py          (Current Progress)
├── bev_encoder.py       （Projection / Geometry）
├── bev_utils.py          (Future)
```

---

## ⚙️ Installation

```bash
git clone https://github.com/yourusername/bev-autonomous-driving.git
cd bev-autonomous-driving

pip install numpy nuscenes-devkit
```

---

## 🚀 Usage

```python
from bev_dataset import BEVAutoDriveDataset

dataset = BEVAutoDriveDataset(
    nusc=nusc,
    is_train=True,
    data_aug_conf=aug_config
)

samples = dataset.samples
resize, dims, crop, flip, rot = dataset.sample_augmentation()
```

---

## 📊 Future Work

* [ ] BEV feature encoder (CNN / Transformer)
* [ ] Multi-camera fusion
* [ ] Integration with MPC / RL control
* [ ] Energy-aware trajectory optimization
* [ ] Real-world dataset validation (AEMO + fleet data)

---

## 🎯 Research Directions

This project is aligned with research topics including:

* Autonomous Driving
* Robotics & Embodied Intelligence
* Optimal Control & MPC
* Intelligent Transportation Systems
* Energy-aware AI systems

---

## 👤 Author

**Yuanzhe (Nikola) Chen**

PhD Applicant (Fall 2027, United States preferred) | Autonomous Driving & Robotics | Control, Planning & Energy Systems | M.Eng (EE) @ UNSW

Research interests:

* Autonomous Driving & Robotics
* Control Theory & Optimal Control
* EV Energy Systems & Optimization

---

## 📬 Contact

Feel free to reach out for collaboration or discussion:

🔗 LinkedIn: www.linkedin.com/in/yuanzhe-chen-6b2158351

---

## ⭐ Acknowledgement

This project is inspired by modern BEV-based autonomous driving frameworks and is developed for **research and educational purposes**.

---

## 📄 License

MIT License
