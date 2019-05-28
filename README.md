# Development repository for my small intro-to-OpenCL project.

It consists of C++ class auto_tensor<double> which should work like a NumPy's NDArray, but it should be able to store the data in an OpenCL device and make tensor-like operations on it, such as elementwise addition, contraction, or gradient if the tensor is interpreted as a discrete approximation to a function.

The auto prefix comes from the idea that it should handle memory automatically. It served as a nice playfield to learn use out-of-order event handling and memory transfer, but defining such a class with such granularity proves cumbersome, and the rules for which operations happen where is blurried. Once I get it working I'll probably mimmick thrust's host_vector and device_vector syntax, and then define the auto_tensor with the help of those and clearer rules.

host_tensor.cpp should be fully functional already.
