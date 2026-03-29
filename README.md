# BEV Autonomous Driving Algorithms

A research-oriented, independently written implementation of core components
for camera-based Bird's Eye View perception.

## What is included

- dataset abstraction for NuScenes-style camera data
- split-aware scene indexing
- multi-camera image acquisition and augmentation
- frustum construction in image space
- camera-to-ego geometric projection
- voxel-based BEV aggregation
- lightweight camera-view and BEV-view encoders

## Project structure

```text
bev-autonomous-driving/
├── bev_dataset.py
├── bev_geometry.py
├── bev_pooling.py
├── bev_encoders.py
├── example_usage.py
└── utils/
    └── image_ops.py
```

## Design principles

This codebase is intentionally organized around reusable research modules rather
than textbook listing order. The implementation focuses on clarity, modularity,
and extensibility for future work in:

- multi-camera BEV perception
- planning and control
- sensor fusion
- energy-aware autonomous driving systems

## Installation

```bash
pip install torch torchvision pillow numpy pyquaternion nuscenes-devkit
```

## Notes

This repository implements common BEV perception ideas using original code and
documentation written for research and educational use.
