# Modularity-based Community Detection algorithms

TODO: short description

#### Usage
Install the dependencies from requirements.txt. As the code for RenEEL is written in C, you also need to have **gcc**.
To run all tests:
```sh
$ python community_detection.py
```
To run tests without including the genetic algorithm (the genetic algorithm is very slow)
```sh
$ python community_detection.py -w
```
After the run, the results can be found in the `results/` in the form of png and json files.
If you wish to see live updates in the terminal, the `VERBOSE` variable can be manually changed to `True` in the code.

#### Acknowledgement
This project made use of the code for the [RenEEL Paper](https://github.com/kbassler/RenEEL-Modularity)
