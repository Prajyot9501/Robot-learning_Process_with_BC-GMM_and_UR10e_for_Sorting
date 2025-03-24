# Robot-learning-524

This repository contains code and resources related to robot learning for sorting problems on a real UR10e robot, focusing on training RL models for robotic tasks using RoboMimic and RoboSuite environments.

**Video Demonstration:**  
[Watch UR10e Sorting Task Demo](./robot_learning_ws/src/robot_learning/data/videos/real_recording_1.mp4)
[Watch UR10 Robosuite picking blocks](./robot_learning_ws/src/robot_learning/data/videos/playback_sim2_yellow.mp4)

*Click the link above to view a demonstration video showcasing the sorting task performed by the UR10e robot.*

---


## Repository Structure

- `.vscode/`: Configuration files for Visual Studio Code.
- `mimicgen/`: Scripts and tools for generating MimicGen data.
- `robomimic/`: RoboMimic framework implementation and utilities.
- `robosuite/`: Robosuite environments and related configurations.
- `robot_learning_ws/src/`: ROS workspace source files for robot learning tasks.
- `.gitignore`: Files/directories to be ignored by Git.
- `pick_and_place_log.csv`: Sample dataset logs for pick-and-place tasks.
- `pick_and_place_log.json`: JSON-formatted logs for pick-and-place tasks.

---

## Prerequisites

Before running the project, ensure you have:

- Ubuntu Linux (22.04 or newer)
- Python 3.8 or higher
- ROS2 Humble
- Git
- Python Virtual Environment (`venv`) or Anaconda (`conda`)
- UR10e

---

## Installation

Follow these steps to install and set up your environment:

### Step 1: Clone the repository
```bash
git clone https://github.com/htella26/Robot-learning-524.git
cd Robot-learning-524
```
### Step 2: Create environments
```bash
python3 -m venv venv
source venv/bin/activate
```
### Step 3: Install required dependencies
```bash
pip install numpy scipy torch torchvision robosuite robomimic open3d scikit-learn
```
### Step 4: Running the code
```bash
cd robot_learning_ws/src/robot_learning
python3 pick_and_place.py

