#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <functional>
#include <cmath>

//Assume elements of a body, since that's somewhat what vectors are useful for
template<typename K>
class host_tensor
{
	private:

	int m_type=0;
	int *m_shape=NULL;
	int m_elems=0;
	K *m_data=NULL;
	bool m_subt=0;

	K m_epsilon=0.0001;
	K m_dx=1;

	//
	//Subthingies

	host_tensor<K> subtensor(int type, int *shape, int elems, K *data) const
	{
		host_tensor<K> ret;
		ret.m_type=type;
		ret.m_shape=shape;
		ret.m_elems=elems;
		ret.m_data=data;
		ret.m_subt=1;
		return ret;
	}

	host_tensor<K> sub_scalar(K *scalar) const
	{
		host_tensor<K> ret;
		ret.m_elems=1;
		ret.m_data=scalar;
		ret.m_subt=1;
		return ret;
	}

	//
	//Initialization helpers
	
	inline void set_shape(const std::vector<int> &shape)
	{
		for(int i=0; i<shape.size(); i++) if(shape[i]<=0) return;
		m_type=shape.size();
		if(m_type==0) {
			m_elems=1;
			m_data=new K[1];
			return;
		}
		m_shape=new int[m_type];
		m_elems=1;
		for(int i=0; i<m_type; i++) m_elems*=(m_shape[i]=shape[i]);
		m_data=new K[m_elems];
	}

	//
	//Elementwise general functions

	//Elementwise operation with scalar
	inline host_tensor<K> ew_operation(const host_tensor<K> &A, const K k, std::function<K(const K, const K)> o) const
	{
		host_tensor<K> ret(A);
		for(int i=0; i<A.m_elems; i++) ret.m_data[i]=o(ret.m_data[i], k);
		return ret;
	}

	//in place element wise operation with scalar
	inline host_tensor<K>& ip_ew_operation(host_tensor<K> &A, const K k, std::function<void(K&, const K)> o)
	{
		for(int i=0; i<A.m_elems; i++) o(A.m_data[i], k);
		return A;
	}

	inline bool same_shape(const host_tensor<K> &A, const host_tensor<K> &B) const
	{
		if(A.m_type!=B.m_type) return 0;
		if(!A.m_type) return A.m_elems==B.m_elems;
		for(int i=0; i<A.m_type; i++) if(A.m_shape[i]!=B.m_shape[i]) return 0;
		return 1;
	}

	//element_wise operation with another tensor
	inline host_tensor<K> ew_operation(const host_tensor<K> &A, const host_tensor<K> &B, std::function<K(const K, const K)> o) const
	{
		if(!same_shape(A, B)) return host_tensor<K>();
		host_tensor<K> ret(A);
		for(int i=0; i<A.m_elems; i++) ret.m_data[i]=o(A.m_data[i], B.m_data[i]);
		return ret;
	}

	//in place element wise operation with another tensor
	inline host_tensor<K>& ip_ew_operation(host_tensor<K> &A, const host_tensor<K> &B, std::function<void(K&, const K)> o)
	{
		if(!same_shape(A, B)) return A=host_tensor<K>();
		for(int i=0; i<A.m_elems; i++) o(A.m_data[i], B.m_data[i]);
		return A;
	}


	//
	//Maybe too much but, operators as functions
	
	static inline void copy(K &x, const K y)
	{
		x=y;
	};

	static inline K addition(const K x, const K y)
	{
		return x+y;
	}

	static inline void ip_addition(K &x, const K y)
	{
		x+=y;
	}

	static inline K substraction(const K x, const K y)
	{
		return x-y;
	}

	static inline void ip_substraction(K &x, const K y)
	{
		x-=y;
	}

	static inline K multiplication(const K x, const K y)
	{
		return x*y;
	}

	static inline void ip_multiplication(K &x, const K y)
	{
		x*=y;
	}

	static inline K division(const K &x, const K y)
	{
		return x/y;
	}

	static inline void ip_division(K &x, const K y)
	{
		x/=y;
	}
	
	inline bool eqs(const K &x, const K &y) const
	{
		return abs(x-y)<m_epsilon;
	}

	public:

	//
	//Constructors

	host_tensor<K>() {}

	host_tensor<K>(const K scalar)
	{
		m_elems=1;
		m_data=new K[1];
		*m_data=scalar;
	}

	host_tensor<K>(const std::vector<int> &shape)
	{
		set_shape(shape);
	}

	host_tensor<K>(const std::vector<int> &shape, K val)
	{
		set_shape(shape);
		ip_ew_operation(*this, val, copy);
	}

	host_tensor<K>(const host_tensor<K> &ten)
	{
		m_type=ten.m_type;
		m_dx=ten.m_dx;
		if(m_type) {
			m_shape=new int[m_type];
			for(int i=0; i<m_type; i++) m_shape[i]=ten.m_shape[i];
		}
		m_elems=ten.m_elems;
		if(m_elems) {
			m_data=new K[m_elems];
			for(int i=0; i<m_elems; i++) m_data[i]=ten.m_data[i];
		}
	}

	~host_tensor<K>()
	{
		if(m_subt) return;
		if(m_shape) delete[] m_shape;
		if(m_data) delete[] m_data;
	}


	//Copy constructors
	host_tensor<K> &operator=(const host_tensor<K> ten)
	{
		m_dx=ten.m_dx;
		if(same_shape(*this, ten)) {
			ip_ew_operation(*this, ten, copy);
			return *this;
		}
		if(m_subt) return *this;
		if(m_type) delete[] m_shape;
		if(m_elems) delete[] m_data;
		if(m_type=ten.m_type) {
			m_shape=new int[m_type];
			for(int i=0; i<m_type; i++) m_shape[i]=ten.m_shape[i];
		} else	m_shape=NULL;
		if(m_elems=ten.m_elems) {
			m_data=new K[m_elems];
			ip_ew_operation(*this, ten, copy);
		} else m_data=NULL;
		return *this;
	}

	host_tensor <K> &operator=(const K scalar)
	{
		if(m_subt) {
			if(!m_type && m_elems) *m_data=scalar;
			else *this=host_tensor<K>();
			return *this;
		}
		if(m_type) delete[] m_shape;
		if(m_elems) delete[] m_data;
		m_type=0;
		m_shape=NULL;
		m_elems=1;
		m_data=new K[1];
		m_subt=0;
		*m_data=scalar;
	}

	host_tensor <K> &operator=(const std::function<K(const std::vector<int>&)> f)
	{
		if(m_type==0) {
			if(m_elems) *m_data=f({});
			return *this;
		}
		std::vector<int> indexes(m_type, 0);
		int m_els;
		for(int i=0; i<m_elems; i++) {
			m_els=m_elems;
			for(int j=0; j<m_type; j++) {
				indexes[j]=m_els/m_shape[j];
				m_els%=m_shape[j];
			}
			m_data[i]=f(indexes);
		}
		return *this;
	}


	//
	//Elementwise operators
	
	host_tensor<K> operator+(const host_tensor<K> &ten) const
	{
		return ew_operation(*this, ten, addition);
	}

	host_tensor<K> operator+(const K k) const
	{
		if(m_type>1) return host_tensor<K>();
		return ew_operation(*this, k, addition);
	}

	template<typename T>
	friend host_tensor<T> operator+(const T k, const host_tensor<T> &ten); //TODO: this

	host_tensor<K> &operator+=(const host_tensor<K> &ten)
	{
		return ip_ew_operation(*this, ten, ip_addition);
	}

	host_tensor<K> &operator+=(const K k)
	{
		return ip_ew_operation(*this, k, ip_addition);
	}

	host_tensor<K> operator-(const host_tensor<K> &ten) const
	{
		return ew_operation(*this, ten, substraction);
	}

	host_tensor<K> operator-(const K k) const
	{
		if(m_type>1) return host_tensor<K>();
		return ew_operation(*this, k, substraction);
	}

	template<typename T>
	friend host_tensor<T> operator-(const T k, const host_tensor<T> &ten); //TODO: this

	host_tensor<K> &operator-=(const host_tensor<K> &ten)
	{
		return ip_ew_operation(*this, ten, ip_substraction);
	}

	host_tensor<K> &operator-=(const K k)
	{
		return ip_ew_operation(*this, k, ip_substraction);
	}

	host_tensor<K> operator*(const host_tensor<K> &ten) const
	{
		return ew_operation(*this, ten, multiplication);
	}

	host_tensor<K> operator*(const K k) const
	{
		if(m_type>1) return host_tensor<K>();
		return ew_operation(*this, k, multiplication);
	}

	template<typename T>
	friend host_tensor<T> operator*(const T k, const host_tensor<T> &ten); //TODO: this

	host_tensor<K> &operator*=(const host_tensor<K> &ten)
	{
		return ip_ew_operation(*this, ten, ip_multiplication);
	}

	host_tensor<K> &operator*=(const K k)
	{
		return ip_ew_operation(*this, k, ip_multiplication);
	}

	host_tensor<K> operator/(const K k) const
	{
		if(m_type>1) return host_tensor<K>();
		return ew_operation(*this, k, division);
	}

	template<typename T>
	friend host_tensor<T> operator/(const T k, const host_tensor<T> &ten); //TODO: this

	host_tensor<K> &operator/=(const host_tensor<K> &ten)
	{
		return ip_ew_operation(*this, ten, ip_division);
	}

	host_tensor<K> &operator/=(const K k)
	{
		return ip_ew_operation(*this, k, ip_division);
	}

	bool operator==(const host_tensor<K> &ten) const
	{
		if(!same_shape(*this, ten)) return 0;
		for(int i=0; i<m_elems; i++) if(!eqs(m_data[i], ten.m_data[i])) return 0;
		return 1;
	}

	bool operator!=(const host_tensor<K> &ten) const
	{
		if(!same_shape(*this, ten)) return 1;
		for(int i=0; i<m_elems; i++) if(!eqs(m_data[i], ten.m_data[i])) return 1;
		return 0;
	}

	
	//
	//Other operators
	
	template<typename T>
	friend std::ostream& operator<<(std::ostream &out, const host_tensor<T> &ten);

	host_tensor<K> operator[](int i) const
	{
		if(!m_type) return host_tensor<K>();
		if(i<-m_shape[0] || i>=m_shape[0]) return host_tensor<K>();
		if(i<0) i+=m_shape[0];
		if(m_type==1) return sub_scalar(m_data+i);
		return subtensor(m_type-1, m_shape+1, m_elems/m_shape[0], m_data+i*m_elems/m_shape[0]);
	}


	//
	//Elementwise operations
	
	template<typename T>
	host_tensor<T> apply(std::function<K(const T)> f)
	{
		host_tensor<T> ret(shape());
		for(int i=0; i<m_elems; i++) ret.m_data[i]=f(m_data[i]);
		return ret;
	}

	host_tensor<K>& substitute(std::function<K(const K)> f)
	{
		for(int i=0; i<m_elems; i++) m_data[i]=f(m_data[i]);
	}

	host_tensor<K> apply(std::function<K(const K)> f)
	{
		host_tensor<K> ret(shape());
		for(int i=0; i<m_elems; i++) ret.m_data[i]=f(m_data[i]);
		return ret;
	}

	host_tensor<K> apply(std::function<K(const K, const std::vector<int> &)> f)
	{
		host_tensor<K> ret(shape());
		std::vector<int> index(m_type, 0);
		for(int i=0; i<m_elems; i++) {
			int step=m_elems, ind=i;
			for(int j=0; j<m_type; j++) {
				step/=m_shape[j];
				index[j]=ind/step%m_shape[j];
				ind%=step;
			}
			ret.m_data[i]=f(m_data[i], index);
		}
		return ret;
	}

	//TODO: this
	//template<typename T>
	//friend host_tensor<T> trace(const host_tensor<T> &A, int i_A, int j_A);

	template<typename T>
	friend host_tensor<T> reduce(const host_tensor<T> &A, int i_A, const host_tensor<T> &B, int i_B);

	template<typename T>
	friend host_tensor<T> derivative(const host_tensor<T> &A, int i_A);

	template<typename T>
	friend host_tensor<T> grad(const host_tensor<T> &A);

	template<typename T>
	friend host_tensor<T> div(const host_tensor<T> &A);

	template<typename T>
	friend host_tensor<T> laplacian(const host_tensor<T> &A);

	//
	//General info functions
	
	bool err()
	{
		return !m_elems && !m_type;
	}

	std::vector<int> shape() const
	{
		return std::vector<int>(m_shape, m_shape+m_type);
	}

	int shape(const int i) const
	{
		if(i<0 || i>=m_type) return -1;
		return m_shape[i];
	}

	int type() const
	{
		return m_type;
	}

	//Extras
	
	//template<typename T>
	//friend host_tensor<T> K_delta(const std::vector<int> &shape);


	//Physics yeah baby
	
	void set_dx(const K dx)
	{
		m_dx=dx;
	}

	K get_dx()
	{
		return m_dx;
	}

};

template<typename T>
host_tensor<T> operator+(const T k, const host_tensor<T> &ten)
{
	return ten.ew_operation(ten, k, [](const T x, const T y)->T { return y+x; });
}

template<typename T>
host_tensor<T> operator-(const T k, const host_tensor<T> &ten)
{
	return ten.ew_operation(ten, k, [](const T x, const T y)->T { return y-x; });
}

template<typename T>
host_tensor<T> operator*(const T k, const host_tensor<T> &ten)
{
	return ten.ew_operation(ten, k, [](const T x, const T y)->T { return y*x; });
}

template<typename T>
host_tensor<T> operator/(const T k, const host_tensor<T> &ten)
{
	return ten.ew_operation(ten, k, [](const T x, const T y)->T { return y/x; });
}

template<typename T>
std::ostream& operator<<(std::ostream &out, const host_tensor<T> &ten)
{
	if(ten.m_type==0) {
		if(ten.m_elems) out << *ten.m_data;
		return out;
	}
	out << "[ ";
	for(int i=0; i<ten.m_shape[0]-1; i++) out << ten.subtensor(ten.m_type-1, ten.m_shape+1, ten.m_elems/ten.m_shape[0], ten.m_data+i*ten.m_elems/ten.m_shape[0]) << ", ";
	out << ten.subtensor(ten.m_type-1, ten.m_shape+1, ten.m_elems/ten.m_shape[0], ten.m_data+(ten.m_shape[0]-1)*ten.m_elems/ten.m_shape[0]) << " ]";
	return out;
}

template<typename T>
host_tensor<T> derivative(const host_tensor<T> &A, int i_A)
{
	if(i_A<0 || i_A>=A.m_type) return host_tensor<T>();
	if(A.m_type==0)
		if(A.m_elems) return host_tensor<T>(0);
		else return host_tensor<T>();
	host_tensor<T> ret(A.shape());
	ret.m_dx=A.m_dx;
	int step=1;
	for(int i=A.m_type-1; i>i_A; i--) step*=A.m_shape[i];
	int N_A=A.m_shape[i_A];
	for(int i=0; i<ret.m_elems; i++) {
		int j=i/step%N_A;
		ret.m_data[i]= 0.5/A.m_dx*(A.m_data[i-step*j+(j+1)%N_A*step]-A.m_data[i-step*j+(j+N_A-1)%N_A*step]);
	}
	return ret; //TODO: ver que esté bien
}

template<typename T>
host_tensor<T> grad(const host_tensor<T> &A)
{
	if(A.m_type==0)
		if(A.m_elems) return host_tensor<T>(0);
		else return host_tensor<T>();
	if(A.m_type==1) {
		host_tensor<T> ret(A.shape());
		ret.m_dx=A.m_dx;
		for(int i=0; i<A.m_elems; i++) ret.m_data[i]= 0.5/A.m_dx*(A.m_data[(i+1)%A.m_elems]-A.m_data[(i+A.m_elems-1)%A.m_elems]);
		return ret;
	}
	std::vector<int> shape=A.shape();
	shape.emplace(shape.begin(), A.m_type);
	host_tensor<T> ret(shape);
	ret.m_dx=A.m_dx;

	int step=A.m_elems;
	for(int i_A=0; i_A<A.m_type; i_A++) {
		int N_A=A.m_shape[i_A];
		step/=N_A;
		for(int i=0; i<A.m_elems; i++) {
			int j=i/step%N_A;
			ret.m_data[i_A*A.m_elems+i]= 0.5/A.m_dx*(A.m_data[i-j*step+(j+1)%N_A*step]-A.m_data[i-j*step+(j+N_A-1)%N_A*step]);
		}
	}
	return ret;
}

template<typename T>
host_tensor<T> div(const host_tensor<T> &A)
{
	if(A.m_type==0)
		if(A.m_elems) return host_tensor<T>(0);
		else return host_tensor<T>();
	if(A.m_shape[0]!=A.m_type-1) return host_tensor<T>();

	std::vector<int> shape=A.shape();
	shape.erase(shape.begin());
	host_tensor<T> ret(shape, 0);
	ret.m_dx=A.m_dx;

	int step=ret.m_elems;
	for(int i_A=0; i_A<ret.m_type; i_A++) {
		int N_A=ret.m_shape[i_A];
		step/=N_A;
		for(int i=0; i<ret.m_elems; i++) {
			int j=i/step%N_A;
			ret.m_data[i]+= 0.5/A.m_dx*(A.m_data[i_A*ret.m_elems+1-j*step+(j+1)%N_A*step]-A.m_data[i_A*ret.m_elems+i-j*step+(j+N_A-1)%N_A*step]);
		}
	}
	return ret;
}

template<typename T>
host_tensor<T> laplacian(const host_tensor<T> &A)
{
	if(A.m_type==0)
		if(A.m_elems) return host_tensor<T>(0);
		else return host_tensor<T>();
	host_tensor<T> ret(A.shape(), 0);
	ret.m_dx=A.m_dx;
	int step=A.m_elems;
	for(int i_A=0; i_A<A.m_type; i_A++) {
		int N_A=A.m_shape[i_A];
		step/=N_A;
		for(int i=0; i<A.m_elems; i++) {
			int j=i/step%N_A;
			ret.m_data[i]+= 1/A.m_dx/A.m_dx*(A.m_data[i-j*step+(j+1)%N_A*step]-2*A.m_data[i]+A.m_data[i-step*j+(j+N_A-1)%N_A*step]);
		}
	}
	return ret;
}

template<typename T>
host_tensor<T> reduce(const host_tensor<T> &A, int i_A, const host_tensor<T> &B, int i_B)
{
	if(i_A<0 || i_A>=A.m_type || i_B<0 || i_B>=B.m_type) return host_tensor<T>();
	//None of them are empty, because 0<=i_A<A.m_type, and so if m_type==0 then i_A cannot satisfy that

	if(A.m_shape[i_A]!=B.m_shape[i_B]) return host_tensor<T>();
	host_tensor<T> ret;
	ret.m_type=A.m_type+B.m_type-2;
	ret.m_shape=new int[ret.m_type];
	ret.m_elems=1;
	for(int i=0; i<i_A; i++) ret.m_elems*=(ret.m_shape[i]=A.m_shape[i]);
	for(int i=i_A+1; i<A.m_type; i++) ret.m_elems*=(ret.m_shape[i]=A.m_shape[i]);
	for(int i=0; i<i_B; i++) ret.m_elems*=(ret.m_shape[i+A.m_type-1]=B.m_shape[i]);
	for(int i=i_B+1; i<B.m_type; i++) ret.m_elems*=(ret.m_shape[i+A.m_type-2]=B.m_shape[i]);
	ret.m_data=new T[ret.m_elems];

	int step_A=1, step_B=1;
	for(int i=i_A+1; i<A.m_type; i++) step_A*=A.m_shape[i];
	for(int i=i_B+1; i<B.m_type; i++) step_B*=B.m_shape[i];
	int steep_A=step_A*A.m_shape[i_A], steep_B=step_B*B.m_shape[i_B];

	for(int i=0; i<ret.m_elems; i++) {
		ret.m_data[i]=0;
		int	n_B=i%(B.m_elems/B.m_shape[i_B]),
			n_A=i/(B.m_elems/B.m_shape[i_B]),
			B_base=n_B/step_B*steep_B+n_B%step_B,
			A_base=n_A/step_A*steep_A+n_A%step_A;
		for(int k=0; k<A.m_shape[i_A]; k++)
			ret.m_data[i]+=A.m_data[A_base+k*step_A]*B.m_data[B_base+k*step_B];
	}

	return ret;
}

template<typename V>
void Euler_step(V &x, double t, double dt, std::function<V(const V&, double)> f)
{
	x+=dt*f(x, t);
}

template<typename V>
void RK4_step(V &x, double t, double dt, std::function<V(const V&, double)> f)
{
	static V k1(x), k2(x), k3(x), k4(x);
	k1=dt*f(x, t);
	k2=dt*f(x+0.5*k1, t+dt/2);
	k3=dt*f(x+0.5*k2, t+dt/2);
	k4=dt*f(x+k3, t+dt);
	x+=1/6.0*k1+1/3.0*k2+1/3.0*k3+k4;
}

int main()
{
	return 0;
}
