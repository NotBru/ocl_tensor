# Development repository for my small intro-to-OpenCL project.

It consists of C++ classes host_tensor<double> and device_tensor<double> which should work like a NumPy's NDArray, but it should be able to store the data in an OpenCL device and make tensor-like operations on it, such as elementwise addition, contraction, or gradient if the tensor is interpreted as a discrete approximation to a function.

This work is discontinued and was made as final project for my Parallel Programming classes. Some details and bug fixes are missing, and redesign would be due.
