# ECE276B_PR1

This is the programming assignment for UCSD ECE 276B: Planning & Learning in Robotics, Project 1: Dynamic Programming

## Installation

```bash
cd starter_code
pip3 install -r requirements.txt
```

## Usage

```
cd starter_code
python3 doorkey.py
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.



# UCSD ECE 276B Project 1: Dynamic Programming

This project will focus on comparing the performance of search-based and sampling-based motion planning algorithms in 3-D Euclidean space. You are provided with a set of 3-D environments described by a rectangular outer boundary and a set of rectangular obstacle blocks. Each rectangle is described by a 9-dimensional vector, specifying its lower left corner ($x_{min}$, $y_{min}$, $z_{min}$), its upper right corner ($x_{max}$, $y_{max}$, $z_{max}$), and its RGB color (for visualization). The start $x_s$ ∈ $R^3$ and goal $x_τ$ ∈ $R^3$ coordinates are also specified for each of the available environments. The provided sample code includes a baseline planner which moves greedily toward the goal. This planner gets stuck in complex environments and is not very careful with collision checking.

<p align="center">
  <img src="https://github.com/homerun-beauty/UCSD-ECE-276B-Project-2-Motion-Planning/assets/60029900/761258aa-20d3-4792-a84c-a8f9ed142cb9" alt="Project Image" width="400">
  <img src="https://github.com/homerun-beauty/UCSD-ECE-276B-Project-2-Motion-Planning/assets/60029900/fda80b7f-5e42-4eb3-a98f-84cf2c133cfb" alt="Project Image" width="400">
</p>
<p align="center">Here is a visual representation of our project in action. The agent is inside the maze environment. </p>

## To get started with the motion planning project, follow these steps:

1. Clone this repository:
  ```bash
  git clone https://github.com/lintsao/UCSD-ECE-276B-Project-1-Dynamic-Programming.git
  cd UCSD-ECE-276B-Project-2-Dynamic-Programming
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


