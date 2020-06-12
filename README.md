# Modularity-based Community Detection algorithms

The following 4 algorithms are part of this experimental study: Clouset-Newman-Moore, Louvain, RenEEL, a genetic algorithm.

#### Usage
Install the dependencies from requirements.txt. As the code for RenEEL is written in C, you also need to have **gcc**.

To run all tests:
```sh
$ python community_detection.py
```
To see possible arguments run:
```sh
$ python community_detection.py -h
```

To activate verbose use:
```sh
$ python community_detection.py -v
```

To set the `mu` parameter to a value (between 0 and 1) run:
```sh
$ python community_detection.py -mu 0.4
```

To run tests without including the genetic algorithm (the genetic algorithm is very slow)
```sh
$ python community_detection.py -w
```

Example run:
```sh
$ python community_detection.py -w -v -mu 0.1
```

After the run, the results can be found in the `results/` in the form of png and json files.

#### Acknowledgement
This project made use of the code for the [RenEEL Paper](https://github.com/kbassler/RenEEL-Modularity)
