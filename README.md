# 1-D Cognitive Processing of Collocated Multi-Gradient Sensing through Layered Sensory Fibers using Neural Networks

This repository will host the code and data associated with our paper:

**Title**: 1-D Cognitive Processing of Collocated Multi-Gradient Sensing through Layered Sensory Fibers using Neural Networks  
**Authors**: 
**Journal**: 
**DOI**: 

## Abstract

This work presents a novel framework for cognitive processing of collocated multi-gradient stimuli using a one-dimensional architecture of layered sensory fibers.

## Dependencies & Installation
Python Version: 3.8+ recommended
```bash
pip install torch pandas numpy matplotlib gradio keyboard tqdm pyserial
```
    

## Data & Assets
### CSV Files:

The training and testing scripts expect CSV files containing sensor resistances (e.g., two columns for two sensors) and a label column (e.g., a key or an action).

Adjust paths in the scripts (e.g., train_test.py, app.py) to point to your local data.

### Images & Other Assets:

The Gradio app (app.py) references images or sounds for the user interface. These are not included in the repository due to large file sizes.

## Training & Testing the Model
### Prepare Your Dataset:

Ensure the CSV files include at least three columns:
1. Inner sensor resistance
2. Outer sensor resistance
3. A label indicating the action/gesture

Modify file paths in train_test.py to match your dataset.

### Run the Training Script:
```bash
python train_test.py
```

## Real-Time UI with Gradio
### app.py:

Launches a Gradio web interface for real-time sensor reading and inference.

Connects to a serial port (e.g., COM22 by default) for live sensor data.

Plots the incoming data and performs inference using the trained model.

Displays the predicted label along with a relevant image (if provided).
```bash
python app.py
```

Please email to request these assets or additional datasets.
