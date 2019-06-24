__kernel
void double_assign(__global double *A, const double k)
{
	A[get_global_id(0)]=k;
}

__kernel
void double_sum(__global double *target, __global double *A, __global double *B)
{
	int idx=get_global_id(0);
	target[idx]=A[idx]+B[idx];
}

__kernel
void double_diff(__global double *target, __global double *A, __global double *B)
{
	int idx=get_global_id(0);
	target[idx]=A[idx]-B[idx];
}

__kernel
void double_prod(__global double *target, __global double *A, __global double *B)
{
	int idx=get_global_id(0);
	target[idx]=A[idx]*B[idx];
}

__kernel
void double_div(__global double *target, __global double *A, __global double *B)
{
	int idx=get_global_id(0);
	target[idx]=A[idx]/B[idx];
}

//The axis' index runs over [0, N]
//step is the size of the increment step for the axis
__kernel
void double_sum_second_deriv(__global double *target, __global double *source, const int step, const int N)
{
	int idx=get_global_id(0);
	int left=idx-step, right=idx+step;
	if(idx/step%N==0) left+=N*step;
	if(idx/step%N==N-1) right-=N*step;
	target[idx]+=source[left]-2*source[idx]+source[right];
}

/*
__kernel
void double_reduce(__global double *target, __global double *source, const int step, const int N, (double *)f(double, double))
{
	//HERE
	int i=get_local_id(0), datasize=(get_global_size(0)-256*get_group_id(0))%256;
	*/

__kernel
void double_ssum(__global double *target, __global double *A, const double k)
{
	int idx=get_global_id(0);
	target[idx]=A[idx]+k;
}

__kernel
void double_sdiff(__global double *target, __global double *A, const double k)
{
	int idx=get_global_id(0);
	target[idx]=A[idx]-k;
}

__kernel
void double_srdiff(__global double *target, __global double *A, const double k)
{
	int idx=get_global_id(0);
	target[idx]=k-A[idx];
}

__kernel
void double_sprod(__global double *target, __global double *A, const double k)
{
	int idx=get_global_id(0);
	target[idx]=A[idx]*k;
}

__kernel
void double_sdiv(__global double *target, __global double *A, const double k)
{
	int idx=get_global_id(0);
	target[idx]=A[idx]/k;
}

__kernel
void double_srdiv(__global double *target, __global double *A, const double k)
{
	int idx=get_global_id(0);
	target[idx]=k/A[idx];
}

__kernel
void to_rgba(__global float *rgba, __global double *source, const double min, const double max)
{
	int i=get_global_id(0);
	double q=2*(source[i]-min)/(max-min)-1;
	i*=4;
	rgba[i+3]=1;
	if(q<=-1) {
		rgba[i+0]=0.12;
		rgba[i+1]=0.00;
		rgba[i+2]=0.31;
	} else if(q>=1) {
		rgba[i+0]=1.00;
		rgba[i+1]=1.00;
		rgba[i+2]=0.19;
	} else if(q<0) {
		rgba[i+0]=-0.12*q;
		rgba[i+1]=0;
		rgba[i+2]=-0.31*q;
	} else if(q<0.5) {
		rgba[i+0]=0.36*q;
		rgba[i+1]=0;
		rgba[i+2]=0;
	} else {
		rgba[i+0]=0.04+0.64*q;
		rgba[i+1]=q-0.5;
		rgba[i+2]=.19*q-.095;
	}
}
