# End-to-end Autopilot for self-driving cars using Tensorflow

An end-to-end system for navigation instead of explicit decomposition of problem into lane marking detection, path planning and control, using images taken from dashcam of a car (66k images)

## Flow
Image (fed with random shift and rotation (aka augmentation)) ---> CNN ---> Driving command (inverse turning radius) 

-5 convolution layers with increasing no. of channels and 3 fully connected layers (with dropout)
-ReLU activations throughout, except last layer where inverse tan activation is used
