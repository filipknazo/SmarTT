# Dynamic Traffic Optimization with AI by SmarTTeam

This repository contains the implementation of our AI-driven solution for dynamic traffic optimization. Developed during a two-day hackathon, this project demonstrates how a Long Short-Term Memory (LSTM) model can be utilized to manage traffic dynamically, using real-world data from smart intersections in the city of Trnava.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Workflow](#workflow)
- [Visualization](#visualization)
- [Contributors](#contributors)

## Overview

Our solution leverages AI to dynamically adjust traffic flow by learning from real-world traffic patterns. The project includes:
- **Data Collection**: Traffic data from 11 smart intersections in Trnava on 26th September.
- **Preprocessing**: Data was cleaned and transformed using `sed` commands.
- **Model Training**: LSTM model trained to predict optimal traffic light adjustments based on historical data.
- **Simulation**: Decision-making capabilities of the model tested in SUMO Eclipse with a generated map of Trnava.
- **Visualization**: Mockups to showcase potential GUI for training and monitoring at https://trnavaflow.surge.sh/. **This is not a core feature, only showcase on how interface for training could look like**. Deployed using surge.

---

## Features
1. **AI-Powered Traffic Management**:
   - Optimization of traffic lights.
   - Predictive analysis using LSTM.
2. **City-Specific Simulation**:
   - Custom Trnava map for realistic testing using SUMO Eclipse.
   
---

## Prerequisites

### Software Requirements
1. **Python 3.12**:
   - Required for data preprocessing, model training, and integration.
2. **SUMO Eclipse 1.21.0**:
   - Used for traffic simulation and testing.
3. **Linux**:
   - We have tested this software in Ubuntu 24

---

## Installation

### Python Libraries
Install the following libraries via pip (recommended in venv):
```bash
pip install numpy pandas tensorflow scikit-learn matplotlib
```

### SUMO Eclipse
Install SUMO Eclipse:
```bash
sudo add-apt-repository ppa:sumo/stable
sudo apt-get update
sudo apt-get install sumo sumo-tools sumo-doc
```

---

## Usage

### Training 
Train the AI model (this might take time):
```bash
python train_model.py
```

### Simulation
Simulate city of Trnava with synthetic data without AI; you have to be in data/sumo_network folder:
```bash
sumo-gui -c osm.sumocfg
```

Simulate city of Trnava with synthetic data with AI.
```commandline
python trnava_simulator.py
```