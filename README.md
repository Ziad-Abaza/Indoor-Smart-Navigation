# Smart City Indoor Localization & Navigation

This repository serves as a core integration for a primitive indoor localization system, developed as part of a Smart City graduation project. The system enables an agent (simulating a vehicle) to navigate on a map toward a designated target point smoothly after determining its current position. A realistic environment parallel to real-world conditions has been built to train the navigation model efficiently.

The system is designed to integrate with an indoor localization solution that uses ESP32 microcontrollers and a Flask server to estimate position using WiFi signal strength. You can find the indoor localization source code [here](https://github.com/algamelomer/indoor-localization.git).

## Overview

The repository contains two primary implementations:

1. **Basic Environment (`smart_city_env.py`)**  
   A primitive simulation of a city grid where roads are defined by a binary map. The agent moves along the grid, collecting points located at predefined "tower" positions. This serves as a proof-of-concept for integration with the localization system.

2. **Advanced Environment with DQN (`smart_city_DQN.py`)**  
   An enhanced version of the simulation environment coupled with a Deep Q-Network (DQN) reinforcement learning agent. This version utilizes noisy linear layers to improve exploration during training. The RL agent learns to navigate the grid toward a dynamically set target point while receiving rewards based on its proximity and correct movements.

## Features

- **Grid-Based Environment:**  
  The city is represented as a 21x21 grid with designated horizontal and vertical road segments.

- **Tower Points:**  
  Specific grid locations represent signal towers (labeled A–E) that are used for localization.

- **Agent Navigation:**  
  The agent can move in four directions (up, down, left, right) with allowed actions depending on its current road segment.

- **Reinforcement Learning:**  
  The advanced version integrates a DQN-based agent with:
  - Noisy layers to enhance exploration.
  - A target network for stability.
  - Experience replay with a large memory buffer.

- **Integration Ready:**  
  Designed as a nucleus for integrating with a real indoor localization system.

## Development Roadmap

This project is intended to evolve into a fully functional autonomous navigation system. Future improvements include:

- **Integration with Indoor Localization:**  
  The environment will be directly linked with the indoor localization system, allowing the agent to receive real-time position updates.

- **Traffic Rules and Priorities:**  
  The environment will be expanded to incorporate traffic rules, right-of-way logic, and priority-based navigation to simulate real-world conditions.

- **Autonomous City Navigation:**  
  The agent will be trained to reach any desired point within the city without requiring pre-defined waypoints. The model will dynamically determine the best path using real-time localization and reinforcement learning.

## Requirements

- Python 3.6+
- [Gym](https://www.gymlibrary.ml/)
- [NumPy](https://numpy.org/)
- [Pygame](https://www.pygame.org/news)
- [PyTorch](https://pytorch.org/) (for the DQN implementation)

Install the required packages using pip:

```bash
pip install gym numpy pygame torch
```

## Usage

### Running the Basic Environment

To run the basic simulation environment:

```bash
python smart_city_env.py
```

This script will:
- Initialize the grid-based environment.
- Randomly position the agent on a road.
- Allow the agent to move randomly until all tower points are visited.
- Render the environment using Pygame.

### Running the Advanced DQN Environment

To train and test the DQN-based agent:

```bash
python smart_city_DQN.py
```

This script will:
- Initialize the enhanced simulation environment.
- Set up a DQN agent with noisy layers.
- Train the agent over multiple episodes while rendering the simulation.
- Save the trained model for later use.
- Run test episodes to demonstrate the agent’s performance.

## Visualization

Below is an example of the environment rendered using Gym:

![Environment Visualization](./screenshots/screenshot.png)

## Contributing

Contributions and feedback are welcome! If you have suggestions or improvements, please feel free to open an issue or submit a pull request.

## License

This project is provided for educational purposes. (Include your license information here if applicable.)

