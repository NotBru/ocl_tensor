#include <iostream>
#include <fstream>
#include <vector>

#ifdef __APPLE__
	#include <OpenCL/cl.hpp>
#else
	#include <CL/cl.hpp>
#endif

template<typename T>
class auto_tensor
{
	private:
		auto_tensor();
	public:
};

template<>
class auto_tensor<double>
{

	private:
	
	static cl::Device s_default_device;
	static cl::Context s_default_context;
	static cl::CommandQueue s_default_queue;
	static cl::Program s_default_program;

	static cl::Kernel s_kern_assign;
	static cl::Kernel s_kern_sum;

	int m_type=0;
	int *m_shape=NULL;
	int m_elems=0;
	double *m_data=NULL;
	cl::Buffer *m_ddata=NULL;

	auto_tensor<double> *m_parent=NULL, *m_first=this;

	double m_dx=1;

	//Event handling
	std::vector<cl::Event*> m_lastaccesses=std::vector<cl::Event*>(4, NULL);
	typedef enum {
		HTD = 1<<0, //1
		DTH = 1<<1, //2
		DEV_READ = 1<<2, //4
		DEV_WRITE = 1<<3, //8
		HAD = HTD | DTH, //3
		DEV_ALL = DEV_READ | DEV_WRITE, //12
		ALL = HAD | DEV_ALL, //15
		WRITE_ALL = DTH | DEV_WRITE, //10
		READ_ALL = HTD | DEV_READ //5
	} EventKind;

	inline int event_pos(int event_kind) const
	{
		//return 0; //WARNING
		if(event_kind==HTD) return 0;
		if(event_kind==DTH) return 1;
		if(event_kind==DEV_READ) return 2;
		if(event_kind==DEV_WRITE) return 3;
		return -1;
	}

	inline cl::Event* event_pointer(int event_kind)
	{
		int i=event_pos(event_kind);
		//i=0; //WARNING
		if(!m_lastaccesses[i]) m_lastaccesses[i]=new cl::Event();
		return m_lastaccesses[i];
	}

	inline bool event_includes(int event_kind, int i) const
	{
		//return event_kind; //WARNING
		if(i==0) return event_kind&HTD;
		if(i==1) return event_kind&DTH;
		if(i==2) return event_kind&DEV_READ;
		if(i==3) return event_kind&DEV_WRITE;
		return 0;
	}

	inline void push_event(int event_kind) const
	{
		int i=event_pos(event_kind);
		if(!m_lastaccesses[i]) return;
		for(auto_tensor<double> *it=m_parent; it; it=it->m_parent) {
			if(!it->m_lastaccesses[i]) it->m_lastaccesses[i]=new cl::Event;
			*it->m_lastaccesses[i]=*m_lastaccesses[i];
		}
	}

	inline std::vector<cl::Event> gen_waitlist(int event_kind = ALL) const
	{
		std::vector<cl::Event> ret;
		for(const auto_tensor<double> *it=this; it; it=it->m_parent)
			for(int i=0; i<4; i++) if(event_includes(event_kind, i) && it->m_lastaccesses[i]) ret.push_back(*it->m_lastaccesses[i]);
		return ret;
	}

	void wait(int access_kind) const
	{
		for(const auto_tensor<double> *it=this; it; it=it->m_parent)
			for(int i=0; i<4; i++) if(event_includes(access_kind, i) && it->m_lastaccesses[i])
				it->m_lastaccesses[i]->wait();
	}

	//Host-side info private handling
	
	inline void reset()
	{
		if(m_ddata) {
			delete m_ddata;
			m_ddata=NULL;
		}
		if(!m_parent && m_data) {
			delete[] m_data;
			m_data=NULL;
		}
		if(m_shape) {
			delete[] m_shape;
			m_shape=NULL;
		}
		for(int i=0; i<4; i++) if(m_lastaccesses[i]) {
			delete m_lastaccesses[i];
			m_lastaccesses[i]=NULL;
		}
		m_elems=0;
		m_type=0;
	}

	inline int elements_for_shape(const std::vector<int> &shape)
	{
		if(shape.size()==0) return 0;
		int ret=1;
		for(int i=0; i<shape.size(); i++) {
			if(shape[i]<=0) return -1;
			ret*=shape[i];
		}
		return ret;
	}

	inline void set_shape(const std::vector<int> &shape)
	{
		for(int i=0; i<shape.size(); i++) if(shape[i]<=0) return;
		m_type=shape.size();
		m_elems=1;
		if(m_type!=0) {
			m_shape=new int[m_type];
			for(int i=0; i<m_type; i++) m_elems*=(m_shape[i]=shape[i]);
		}
	}

	inline bool same_shape(const auto_tensor<double> &A, const auto_tensor<double> &B) const
	{
		if(A.m_elems!=B.m_elems) return 0;
		if(A.m_type!=B.m_type) return 0;
		for(int i=0; i<A.m_type; i++) if(A.m_shape[i]!=B.m_shape[i]) return 0;
		return 1;
	}

	inline void copy_shape(const auto_tensor<double> &rvalue)
	{
		if(m_type!=rvalue.m_type) {
			m_type=rvalue.m_type;
			if(m_shape)
				delete[] m_shape;
			if(!m_type) m_shape=NULL;
			else m_shape=new int[m_type];
		}
		for(int i=0; i<m_type; i++) m_shape[i]=rvalue.m_shape[i];
	}

	inline void alloc(const double host=1, const double *data=NULL, const int N=1)
	{
		if(host) {
			m_data=new double[m_elems];
			if(data) for(int i=0; i<N; i++) m_data[i]=data[i];
		} else {
			m_ddata=new cl::Buffer(s_default_context, CL_MEM_READ_WRITE, m_elems*sizeof(double));
			if(data) s_default_queue.enqueueWriteBuffer(*m_ddata, CL_FALSE, 0, N*sizeof(double), data, NULL, event_pointer( DEV_WRITE ));
		}
	}

	inline auto_tensor<double> subscalar(const int i)
	{
		cl_buffer_region reg={i*sizeof(double), sizeof(double)};
		auto_tensor<double> ret;
		ret.m_elems=1;
		ret.m_data=m_data+i;
		ret.m_parent=this;
		ret.m_first=m_first;

		if(m_ddata) {
			ret.m_ddata=new cl::Buffer;
			*ret.m_ddata=m_ddata->createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &reg, NULL);
		}

		return ret;
	}

	inline auto_tensor<double> subtensor(const int i)
	{
		int elems=m_elems/m_shape[0];
		cl_buffer_region reg={i*sizeof(double)*elems, sizeof(double)*elems};
		auto_tensor<double> ret;
		ret.m_type=m_type-1;
		ret.m_shape=elems>1?m_shape+1:NULL;
		ret.m_elems=elems;
		if(m_data) ret.m_data=m_data+i*elems;
		ret.m_parent=this;
		ret.m_first=ret.m_first;

		if(m_ddata) {
			ret.m_ddata=new cl::Buffer;
			*ret.m_ddata=m_ddata->createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &reg, NULL);
		}
		return ret;
	}

	public:

	//Constructors
	
	auto_tensor<double>() {}

	auto_tensor<double>(const double scalar, bool host=1)
	{
		m_elems=1;
		alloc(host, &scalar, 1);
	}

	auto_tensor<double>(const std::vector<int> &shape, bool host=1)
	{
		set_shape(shape);
		alloc(host);
	}

	auto_tensor<double>(const std::vector<int> &shape, const double val, const bool host=1)
	{
		set_shape(shape);
		alloc(host);
		if(host) for(int i=0; i<m_elems; i++) m_data[i]=val;
		else {
			//return; //WARNING
			//TODO: Fix
			s_kern_assign.setArg(0, *m_ddata);
			s_kern_assign.setArg(1, &val);
			cl::NDRange global(m_elems);
			cl::NDRange local(256);

			s_default_queue.enqueueNDRangeKernel(s_kern_assign, cl::NullRange, global, local, NULL, event_pointer( DEV_WRITE ));
		}
	}

	auto_tensor<double>(auto_tensor<double> &&tensor)
	{
		m_dx=tensor.m_dx;
		m_elems=tensor.m_elems;
		copy_shape(tensor);
		if(tensor.m_ddata) {
			m_ddata=new cl::Buffer(s_default_context, CL_MEM_READ_WRITE, m_elems*sizeof(double));
			std::vector<cl::Event> waitlist=tensor.gen_waitlist( HTD | DEV_WRITE );
			s_default_queue.enqueueCopyBuffer(*tensor.m_ddata, *m_ddata, 0, 0, m_elems*sizeof(double), &waitlist, event_pointer( DEV_WRITE ));
			event_pointer( DEV_WRITE )->wait();
		} else alloc(1, tensor.m_data, m_elems);
	}

	~auto_tensor<double>()
	{
		if(m_ddata) delete m_ddata;
		for(int i=0; i<4; i++) delete m_lastaccesses[i];
		if(m_parent) return;
		if(m_data) delete[] m_data;
		if(m_shape) delete[] m_shape;
	}

	class _init
	{
		public:

		_init()
		{
			std::vector<cl::Platform> all_platforms;
			cl::Platform::get(&all_platforms);

			if(all_platforms.size()==0) {
				std::cerr << "No platforms found. Check OpenCL Installation.\n";
				exit(1);
			}

			std::vector<cl::Device> devices;
			for(int i=0; i<all_platforms.size(); i++)
				all_platforms[i].getDevices(CL_DEVICE_TYPE_GPU, &devices);

			if(devices.size()==0) {
				for(int i=0; i<all_platforms.size(); i++)
					all_platforms[i].getDevices(CL_DEVICE_TYPE_CPU, &devices);
			}

			auto_tensor<double>::s_default_device=devices[0];
			auto_tensor<double>::s_default_context=cl::Context(auto_tensor<double>::s_default_device);
			auto_tensor<double>::s_default_queue=cl::CommandQueue(auto_tensor<double>::s_default_context, auto_tensor<double>::s_default_device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);

			std::ifstream inf("auto_tensor.cl");
			std::string sourcestr((std::istreambuf_iterator<char>(inf)), (std::istreambuf_iterator<char>()));
			cl::Program::Sources source(1, std::make_pair(sourcestr.c_str(), sourcestr.length()+1));
			auto_tensor<double>::s_default_program=cl::Program(auto_tensor<double>::s_default_context, source);

			if(auto_tensor<double>::s_default_program.build({auto_tensor<double>::s_default_device}) != CL_SUCCESS) {
				std::cerr << "Error building \"auto_tensor.cl\" source file:\n";
				std::cerr << auto_tensor<double>::s_default_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(auto_tensor<double>::s_default_device) << "\n";
			}

			auto_tensor<double>::s_kern_assign=cl::Kernel(s_default_program, "double_assign");
			auto_tensor<double>::s_kern_sum=cl::Kernel(s_default_program, "double_sum");

		}
	};

	private: static _init s_initializer;

	public:

	//Host-side info public handling
	
	std::vector<int> shape()
	{
		return std::vector<int>(m_shape, m_shape+m_type);
	}

	void reshape(const std::vector<int> shape)
	{
		if(elements_for_shape(shape)!=m_elems) return;
		set_shape(shape);
	}

	friend std::ostream &operator<<(std::ostream &out, auto_tensor<double> &tensor);

	//Host-device information flow public handling

	void push()
	{
		if(!m_elems) return;
		if(!m_ddata) m_ddata=new cl::Buffer(s_default_context, CL_MEM_READ_WRITE, m_elems*sizeof(double));
		else if(m_ddata->getInfo<CL_MEM_SIZE>()!=m_elems*sizeof(double)) *m_ddata=cl::Buffer(s_default_context, CL_MEM_READ_WRITE, m_elems*sizeof(double));

		std::vector<cl::Event> waitlist=gen_waitlist( ALL ); //TODO: add error handling to every enqueue call
		s_default_queue.enqueueWriteBuffer(*m_ddata, CL_FALSE, 0, m_elems*sizeof(double), m_data, &waitlist, event_pointer( HTD ));
		push_event( HTD );
	}

	void pull()
	{
		if(!m_elems) return;
		if(!m_ddata || m_ddata->getInfo<CL_MEM_SIZE>()!=m_elems*sizeof(double)) {
			reset();
			return;
		}
		if(!m_data) m_data=new double[m_elems];

		std::vector<cl::Event> waitlist=gen_waitlist( ALL );
		s_default_queue.enqueueReadBuffer(*m_ddata, CL_FALSE, 0, m_elems*sizeof(double), m_data, &waitlist, event_pointer( DTH ));
		push_event( DTH );
	}

	auto_tensor<double> operator[](int i)
	{
		if(!m_type) return auto_tensor<double>();
		if(i<-m_shape[0] || i>=m_shape[0]) return auto_tensor<double>();
		if(i<0) i+=m_shape[0];
		return subtensor(i);
	}

	auto_tensor<double> &operator=(const double scalar)
	{
		wait( HAD );
		if(m_elems==0) alloc(1, &scalar, 1);
		else if(m_elems==1)
			if(m_data) *m_data=scalar;
			else {
				std::vector<cl::Event> waitlist=gen_waitlist( ALL );
				s_default_queue.enqueueWriteBuffer(*m_ddata, CL_FALSE, 0, sizeof(double), &scalar, &waitlist, event_pointer( HTD ));
				push_event( HTD );
			}
		else {
			reset();
			if(!m_parent) {
				m_elems=1;
				alloc(1, &scalar, 1);
			}
		}
		return *this;
	}

	auto_tensor<double> &operator=(auto_tensor<double> &ten)
	{
		if(m_parent) {
			if(m_elems && m_elems==ten.m_elems && same_shape(*this, ten)) {
				if(m_ddata) {
					if(ten.m_ddata) {
						std::vector<cl::Event> waitlist=gen_waitlist( ALL ), temp=ten.gen_waitlist( HTD | DEV_WRITE );
						waitlist.insert(waitlist.end(), temp.begin(), temp.end());
						s_default_queue.enqueueCopyBuffer(*ten.m_ddata, *m_ddata, 0, 0, m_elems*sizeof(double), &waitlist, event_pointer( DEV_WRITE ));
						push_event( DEV_WRITE );
						*ten.event_pointer( DEV_READ )=*m_lastaccesses[event_pos( DEV_WRITE )];
						ten.push_event( DEV_READ );
					} else {
						std::vector<cl::Event> waitlist=gen_waitlist( ALL );
						s_default_queue.enqueueWriteBuffer(*m_ddata, CL_FALSE, 0, m_elems*sizeof(double), ten.m_data, &waitlist, event_pointer( DEV_WRITE ));
						push_event( DEV_WRITE );
						*ten.event_pointer( HTD )=*m_lastaccesses[event_pos( DEV_WRITE )];
						ten.push_event( HTD );
					}
				} else {
					if(ten.m_ddata)  {
						std::vector<cl::Event> waitlist=ten.gen_waitlist( HTD | DEV_WRITE );
						s_default_queue.enqueueReadBuffer(*ten.m_ddata, CL_FALSE, 0, m_elems*sizeof(double), m_data, &waitlist, ten.event_pointer( DEV_READ ));
						ten.push_event( DEV_READ );
						*event_pointer( DTH )=*ten.m_lastaccesses[event_pos( DEV_READ )];
						push_event( DTH );
					} else for(int i=0; i<m_elems; i++) m_data[i]=ten.m_data[i];
				}
			} else reset();
		} else {
			if(m_elems!=ten.m_elems) {
				reset();
				m_elems=ten.m_elems;
			}
			if(ten.m_ddata) {
				if(!m_ddata) m_ddata=new cl::Buffer(s_default_context, CL_MEM_READ_WRITE, m_elems*sizeof(double));
				std::vector<cl::Event> waitlist=gen_waitlist( ALL ), temp=ten.gen_waitlist( HTD | DEV_WRITE );
				waitlist.insert(waitlist.end(), temp.begin(), temp.end());
				s_default_queue.enqueueCopyBuffer(*ten.m_ddata, *m_ddata, 0, 0, m_elems*sizeof(double), &waitlist, event_pointer( DEV_WRITE ));
				*ten.event_pointer( DEV_READ )=*m_lastaccesses[event_pos( DEV_WRITE )];
				ten.push_event( DEV_READ );
			} else {
				if(m_ddata) {
					std::vector<cl::Event> waitlist=gen_waitlist( ALL );
					s_default_queue.enqueueWriteBuffer(*m_ddata, CL_FALSE, 0, m_elems*sizeof(double), ten.m_data, &waitlist, event_pointer( DEV_WRITE ));
					*ten.event_pointer( HTD )=*m_lastaccesses[event_pos( DEV_WRITE )];
					ten.push_event( HTD );
				} else for(int i=0; i<m_elems; i++) m_data[i]=ten.m_data[i];
			}
			copy_shape(ten);
		}
		return *this;
	}

	auto_tensor<double> &operator=(auto_tensor<double> &&ten) //TODO: optimize for this
	{
		if(m_parent) {
			if(m_elems && m_elems==ten.m_elems && same_shape(*this, ten)) {
				if(m_ddata) {
					if(ten.m_ddata) {
						std::vector<cl::Event> waitlist=gen_waitlist( ALL ), temp=ten.gen_waitlist( HTD | DEV_WRITE );
						waitlist.insert(waitlist.end(), temp.begin(), temp.end());
						s_default_queue.enqueueCopyBuffer(*ten.m_ddata, *m_ddata, 0, 0, m_elems*sizeof(double), &waitlist, event_pointer( DEV_WRITE ));
						event_pointer( DEV_WRITE )->wait();
					} else {
						std::vector<cl::Event> waitlist=gen_waitlist( ALL );
						s_default_queue.enqueueWriteBuffer(*m_ddata, CL_FALSE, 0, m_elems*sizeof(double), ten.m_data, &waitlist, event_pointer( DEV_WRITE ));
						push_event( DEV_WRITE );
					}
				} else {
					if(ten.m_ddata)  {
						std::vector<cl::Event> waitlist=ten.gen_waitlist( HTD | DEV_WRITE );
						s_default_queue.enqueueReadBuffer(*ten.m_ddata, CL_TRUE, 0, m_elems*sizeof(double), m_data, &waitlist);
					} else for(int i=0; i<m_elems; i++) m_data[i]=ten.m_data[i];
				}
			} else reset();
		} else {
			if(m_elems!=ten.m_elems) {
				reset();
				m_elems=ten.m_elems;
			}
			if(ten.m_ddata) {
				if(!m_ddata) m_ddata=new cl::Buffer(s_default_context, CL_MEM_READ_WRITE, m_elems*sizeof(double));
				std::vector<cl::Event> waitlist=gen_waitlist( ALL ), temp=ten.gen_waitlist( HTD | DEV_WRITE );
				waitlist.insert(waitlist.end(), temp.begin(), temp.end());
				s_default_queue.enqueueCopyBuffer(*ten.m_ddata, *m_ddata, 0, 0, m_elems*sizeof(double), &waitlist, event_pointer( DEV_WRITE ));
				event_pointer( DEV_WRITE )->wait();
			} else {
				if(m_ddata) {
					std::vector<cl::Event> waitlist=gen_waitlist( ALL );
					s_default_queue.enqueueWriteBuffer(*m_ddata, CL_FALSE, 0, m_elems*sizeof(double), ten.m_data, &waitlist, event_pointer( DEV_WRITE ));
					event_pointer( DEV_WRITE )->wait();
				} else for(int i=0; i<m_elems; i++) m_data[i]=ten.m_data[i];
			}
			copy_shape(ten);
		}
		return *this;
	}

	//Airthmetic operations
	auto_tensor<double> operator+(auto_tensor<double> &ten)
	{
		if(!same_shape(*this, ten)) return auto_tensor<double>();
		if(m_ddata || ten.m_ddata) {
			if(!ten.m_ddata && !ten.m_parent) ten.push();
			if(!m_ddata && !m_parent) push();

			auto_tensor<double> ret(shape(), (bool)0);
			s_kern_sum.setArg(0, ret.m_ddata);
			if(m_ddata && ten.m_ddata) {
				s_kern_sum.setArg(1, m_ddata);
				s_kern_sum.setArg(2, ten.m_ddata);
			}
			else {
				s_kern_sum.setArg(1, ret.m_ddata);
				if(m_ddata) {
					ret=ten;
					s_kern_sum.setArg(2, m_ddata);
				} else {
					ret=*this;
					s_kern_sum.setArg(2, ten.m_ddata);
				}
			}
	
			cl::NDRange global(m_elems);
			cl::NDRange local(256);
	
			std::vector<cl::Event> waitlist=gen_waitlist( HTD | DEV_WRITE ), temp=ten.gen_waitlist( HTD | DEV_WRITE );
			waitlist.insert(waitlist.end(), temp.begin(), temp.end());
	
			s_default_queue.enqueueNDRangeKernel(s_kern_sum, cl::NullRange, global, local, &waitlist, event_pointer( DEV_READ ));
		
			push_event( DEV_READ);
			*ten.event_pointer( DEV_READ )=*m_lastaccesses[event_pos( DEV_READ )];
			ten.push_event( DEV_READ );
			*ret.event_pointer( DEV_WRITE )=*m_lastaccesses[event_pos( DEV_READ )];
			s_default_queue.finish();
			return ret;
		} else {
			auto_tensor<double> ret(shape());
			for(int i=0; i<m_elems; i++) ret.m_data[i]=m_data[i]+ten.m_data[i];
			return ret;
		}
		return auto_tensor<double>();
	}

};

cl::Device auto_tensor<double>::s_default_device;
cl::Context auto_tensor<double>::s_default_context;
cl::CommandQueue auto_tensor<double>::s_default_queue;
cl::Program auto_tensor<double>::s_default_program;

auto_tensor<double>::_init auto_tensor<double>::s_initializer;

cl::Kernel auto_tensor<double>::s_kern_assign;
cl::Kernel auto_tensor<double>::s_kern_sum;

std::ostream &operator<<(std::ostream &out, auto_tensor<double> &tensor)
{
	if(tensor.m_elems && !tensor.m_data) return out;
	if(tensor.m_type==0 && tensor.m_elems) {
		tensor.wait( tensor.DTH );
		out << *tensor.m_data;
	} else {
		out << "[ ";
		for(int i=0; i<tensor.m_shape[0]-1; i++) {
			auto_tensor<double> subt=tensor[i];
			out << subt << ", ";
		}
		auto_tensor<double> subt=tensor[-1];
		out << subt << " ]";
	}
	return out;
}

int main()
{
	//Playing around without use of kernels
	if(0) {
	std::cout << "auto_tensor A({2, 2}, 3.14)\n";
	auto_tensor<double> A({2, 2}, 3.14);
	std::cout << "A: " << A << "\n";
	std::cout << "A.push()\n";
	A.push();
	std::cout << "A[0][1]=1\n";
	A[0][1]=1;
	std::cout << "A: " << A << "\n";
	std::cout << "A.pull()\n";
	A.pull();
	std::cout << "A: " << A << "\n";
	std::cout << "auto_tensor B({2, 2}, 1.00)\n";
	auto_tensor<double> B({2, 2}, 1.0);
	std::cout << "B: " << B << "\n";
	std::cout << "A=B\n";
	A=B;
	std::cout << "A: " << A << "\n";
	std::cout << "A.pull()\n";
	A.pull();
	std::cout << "A: " << A << "\n";
	std::cout << "A[0][0]=0\n";
	A[0][0]=0;
	std::cout << "A: " << A << "\n";
	std::cout << "B=A\n";
	B=A;
	std::cout << "B: " << B << "\n";
	std::cout << "B.pull()\n";
	B.pull();
	std::cout << "B: " << B << "\n";
	std::cout << "A.push()\n";
	A.push();
	std::cout << "B=A\n";
	B=A;
	std::cout << "B: " << B << "\n";
	std::cout << "B.pull()\n";
	B.pull();
	std::cout << "B: " << B << "\n";
	std::cout << "auto_tensor C({2, 2}, (bool)0)\n";
	auto_tensor<double> C({2, 2}, (bool)0);
	std::cout << "C: " << C << "\n";
	std::cout << "C=B\n";
	C=B;
	std::cout << "C.pull()\n";
	C.pull();
	std::cout << "C: " << C << "\n";
	std::cout << "A=auto_tensor ({2, 2}, 0.577)\n";
	A=auto_tensor<double>({2, 2}, 0.577);
	std::cout << "A: " << A << "\n";
	std::cout << "A.pull()\n";
	A.pull();
	std::cout << "A: " << A << "\n";
	std::cout << "auto_tensor D({2, 2}, 0.577)\n";
	auto_tensor<double> D({2, 2}, 0.577);
	std::cout << "D: " << D << "\n";
	std::cout << "A=D\n";
	A=D;
	std::cout << "A: " << A << "\n";
	std::cout << "A.pull()\n";
	A.pull();
	std::cout << "A: " << A << "\n";
	std::cout << "C[0]=A[0]\n";
	C[0]=A[0];
	std::cout << "C: " << C << "\n";
	std::cout << "C.pull()\n";
	C.pull();
	std::cout << "C: " << C << "\n";
	}


	if(1) {
		/*
	std::cout << "auto_tensor A({16, 16}, 23, 0)\n";
	auto_tensor<double> A({16, 16}, 23, 0);
	std::cout << "A.pull()\n";
	A.pull();
	std::cout << "A: " << A << "\n";
		*/
	std::cout << "auto_tensor A({16, 16}, 0.577)\n";
	auto_tensor<double> A({16, 16}, 0.577);
	std::cout << "A.push()\n";
	A.push();
	std::cout << "auto_tensor B({16, 16}, 2.72)\n";
	auto_tensor<double> B({16, 16}, 2.72);
	std::cout << "auto_tensor C({16, 16})\n";
	auto_tensor<double> C(std::vector<int>{16, 16});
	std::cout << "C=A+B\n";
	C=A+B;
	std::cout << "C: " << C << "\n";
	std::cout << "C.pull()\n";
	C.pull();
	std::cout << "C: " << C << "\n";
	}
	return 0;
}
