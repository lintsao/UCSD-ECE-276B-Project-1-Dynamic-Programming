# UCSD ECE 276B Project 1: Dynamic Programming

This project focuses on autonomous navigation in a Door & Key environment. The objective is to get our agent (red triangle) to the goal location (green square). The environment may contain a door that blocks the way to the goal. If the door is closed, the agent needs to pick up a key to unlock the door. The agent has three regular actions, move forward (MF), turn left (TL), and turn right (TR), and two special actions, pick up the key (PK) and unlock the door (UD). 

<p align="center">
  <img src="https://github.com/lintsao/UCSD-ECE-276B-Project-1-Dynamic-Programming/blob/master/starter_code/gif/DoorKey-8x8-10.gif?raw=true" alt="Project Image" width="400">
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

## License
 
The MIT License (MIT)

Copyright (c) 2015 Chris Kibble

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


