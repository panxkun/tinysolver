### TinySolver: Visual-Inertial Bundle Adjustment Implementation

This repository contains the implementation of a Visual-Inertial Bundle Adjustment (VIO) algorithm. The algorithm is trying to replace the general Bundle Adjustment (BA) algorithm such as [ceres-solver](https://github.com/ceres-solver/ceres-solver) or [g2o](https://github.com/RainerKuemmerle/g2o).


Now this repository is still under development. Next step is to totally comapre it with more other implementations (g2o, gtsam, et.al)

The version with VIO capability will not release but you can try to implement it by yourself. The VIO capability is based on the [XRSLAM](https://github.com/openxrlab/xrslam)


### Performance

Currently, I have conducted an initial comparison with the version of the optimizer implemented using Ceres in a VIO system. On the public EuRoC dataset, it achieves slightly better accuracy and takes only half the computation time compared to Ceres-solver.

