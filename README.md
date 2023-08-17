# UCSD ECE 276B Project 1: Dynamic Programming

This project focuses on autonomous navigation in a Door & Key environment. The objective is to get our agent (red triangle) to the goal location (green square). The environment may contain a door that blocks the way to the goal. If the door is closed, the agent needs to pick up a key to unlock the door. The agent has three regular actions, move forward (MF), turn left (TL), and turn right (TR), and two special actions, pick up the key (PK) and unlock the door (UD). 

<p align="center">
  <img src="https://github.com/lintsao/UCSD-ECE-276B-Project-1-Dynamic-Programming/blob/master/starter_code/gif/DoorKey-8x8-4.gif?raw=true" alt="Project Image" width="400">
<p align="center">Here is a visual representation of our project in action. The goal of the agent is to reach the green grid. </p>

## To get started with the dynamic programming project, follow these steps:

1. Clone this repository:
  ```bash
  git clone https://github.com/lintsao/UCSD-ECE-276B-Project-1-Dynamic-Programming.git
  cd UCSD-ECE-276B-Project-1-Dynamic-Programming
  ```

2. Create a new virtual environment:
  ```bash
  python3 -m venv env
  source env/bin/activate  # For Unix/Linux
  ```

3. Install the required dependencies:
  ```bash
  pip3 install -r requirements.txt
  ```

4. You're ready to use the dynamic programming project!

## Usage

```
cd starter_code
python3 doorkey.py
```

or you could use **doorkey.ipynb** to check the step-by-step implementation.
## Contributing
Contributions are welcome! If you have any suggestions, bug reports, or feature requests, please open an issue or submit a pull request.
