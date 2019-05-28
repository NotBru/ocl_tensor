__kernel
void double_assign(__global double *A, double k)
{
	int idx=get_global_id(0);
	A[idx]=k;
}

__kernel
void double_sum(__global double *target, __global double *A, __global double *B)
{
	int idx=get_global_id(0);
	target[idx]=A[idx]+B[idx];
}
