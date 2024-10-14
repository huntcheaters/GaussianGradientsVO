# GaussianGradientsVO
This is a direct visual odometry for RGB-D cameras. The main contribution of this project is a plain point selection strategy via low order Gaussian derivative functions including Gaussian Gradient Magnitudes (GGM) and Laplacian of Gaussian (LOG) operators. The reference of these metrics is from a Blind Image Quality Assessment (BIQA) approach which is available at https://ieeexplore.ieee.org/abstract/document/6894197. By employing these isotropic metrics locally, we extends possibilities to select salient points while keeping low time consumption and benefiting continuous tracking especially for scenarios lacking in structure and textture. More than that, local selection also broadens searching space for optimizations. In this case, the runtime performance is further improved.
The point selection procedures are from EdgeDirectVO which can be found at https://github.com/kevinchristensen1/EdgeDirectVO.

Searching by title "RGB-D Visual Odometry via Low Order Gaussian Gradient Metrics" would returns the text presentation of this project we have uploaded to TechRxiv.org. 

# Usage of the project
1. Download the source code to local hard disk.
2. Make a directory named build to configure it with cmake, for which the OpenCV I use is 3.4.3.
3. Open the project with an IDE, for which what I am using is Visual Studio 2013.
4. Set GMVO as the startup project at first.
5. Right click on GMVO project name to set the command line arguments to the absolute path of tum1.yaml.
6. Before running it, check and modify the dataset path in tum1.yaml to your local path.

tum1.yaml is the way ORBSLAMs configure the project, and I simply use it in this project. Some setting details in this file is still kept and can be removed as you wish.

# Evaluation with TUM tools
Here is an example
G:\eval_tum>evaluate_rpe.py e:\slamData\rgbd_dataset_freiburg1_rpy\ggmvo.txt e:\slamData\rgbd_dataset_freiburg1_rpy\groundtruth.txt --verbose
Do it for ATE in the same way.

# To cite 
Yao Zhigang et al. Direct RGB-D visual odometry with point features. Intelligent Service Robotics, 2024: 1-13.

Yao Zhigang et al. RGB-D Visual Odometry via Low Order Gaussian Gradient Metrics. TechRxiv. Preprint. https://doi.org/10.36227/techrxiv.22132544.v1
