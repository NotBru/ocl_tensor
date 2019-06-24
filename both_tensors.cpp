#include <iostream>
#include <fstream>
#include <vector>
#include <functional> //For host tensor's mechanisms
#include <map>

#include <cmath>
#define PI 3.141592653589793238

#ifdef __APPLE__
	#include <OpenCL/cl.hpp>
#else
	#include <CL/cl.hpp>
#endif

namespace bru {

	/*
	##############################################################
	##############################################################
			DEVICE TENSOR THINGIES
	##############################################################
	##############################################################
	*/

	namespace { //Private namespace
		using namespace std;
		using namespace cl;

		vector< Platform > platforms;
		vector< vector<Device> > devices;

		vector< Context > contexts;
		vector< Program* > programs;
		vector< vector<CommandQueue> >queues;
		vector< map <string, Kernel> > kernels;

		vector< bool > platform_init;
		vector< vector<bool> > device_init;

		int default_dev[2]={ -1 };

		Program::Sources source;

		class _init
		{
			public:
	
			_init()
			{

				Platform::get(&platforms);
	
				if(platforms.size()==0) {
					std::cerr << "No platforms found. Check OpenCL Installation.\n";
					exit(1);
				}
	
				devices=vector< vector<Device> >(platforms.size());
				contexts=vector<Context>(platforms.size());
				programs=vector< Program* >(platforms.size(), NULL );
				queues=vector< vector<CommandQueue> >(platforms.size());
				kernels=vector< map<string, Kernel> >(platforms.size());
				platform_init=vector< bool >(platforms.size(), 0);
				device_init=vector< vector<bool> >(platforms.size());

				for(int i=0; i<platforms.size(); i++) {
					platforms[i].getDevices(CL_DEVICE_TYPE_GPU, &devices[i]);
					queues[i]=vector<CommandQueue>(devices[i].size());
					device_init[i]=vector<bool>(devices[i].size(), 0);
				}

				//Device selection: picks first GPU found. Should be smarter.
				for(int i=0; i<platforms.size() && default_dev[0]==-1 ; i++)
					for(int j=0; j<devices[i].size(); j++)
						if(devices[i][j].getInfo<CL_DEVICE_TYPE>()==CL_DEVICE_TYPE_GPU) {
							default_dev[0]=i; default_dev[1]=j; }
				if(default_dev[0]==-1)
					for(int i=0; i<platforms.size() && default_dev[0]==-1; i++)
						for(int j=0; j<devices[i].size(); j++)
							if(devices[i][j].getInfo<CL_DEVICE_TYPE>()==CL_DEVICE_TYPE_CPU) {
								default_dev[0]=i; default_dev[1]=j; }
				if(default_dev[0]==-1) {
					default_dev[0]=0; default_dev[1]=0; }

				cl_int err;
				contexts[default_dev[0]]=Context(devices[default_dev[0]], NULL, NULL, NULL, &err);
				if(err != CL_SUCCESS) {
					cerr << "Couldn't instantiate context\n";
					exit(1);
				}

				queues[default_dev[0]][default_dev[1]]=CommandQueue(contexts[default_dev[0]], devices[default_dev[0]][default_dev[1]], CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);
				if(err != CL_SUCCESS) {
					cerr << "Couldn't instantiate queue\n";
					exit(1);
				}
	
				ifstream inf("device_tensor.cl");
				string sourcestr((istreambuf_iterator<char>(inf)), (istreambuf_iterator<char>()));
				source=cl::Program::Sources(1, make_pair(sourcestr.c_str(), sourcestr.length()+1));

				programs[default_dev[0]]=new Program(contexts[default_dev[0]], source); //TODO: delete this in ~_init

				if(programs[default_dev[0]]->build(devices[default_dev[0]], "-cl-std=CL2.0") != CL_SUCCESS) {
					cerr << "Error building \"device_tensor.cl\" source file:\n";
					cerr << programs[default_dev[0]]->getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[default_dev[0]][default_dev[1]]);
					exit(1);
				}

				//Kernels
				//TODO: replace double by TYPE in '.cl' and have an _init for each class specialization compile the code
				//Also, store such functions' names in list
				kernels[default_dev[0]]["double_assign"]=Kernel(*programs[default_dev[0]], "double_assign");
				kernels[default_dev[0]]["double_sum"]=Kernel(*programs[default_dev[0]], "double_sum");
				kernels[default_dev[0]]["double_diff"]=Kernel(*programs[default_dev[0]], "double_diff");
				kernels[default_dev[0]]["double_prod"]=Kernel(*programs[default_dev[0]], "double_prod");
				kernels[default_dev[0]]["double_div"]=Kernel(*programs[default_dev[0]], "double_div");
				kernels[default_dev[0]]["double_ssum"]=Kernel(*programs[default_dev[0]], "double_ssum");
				kernels[default_dev[0]]["double_sdiff"]=Kernel(*programs[default_dev[0]], "double_sdiff");
				kernels[default_dev[0]]["double_sprod"]=Kernel(*programs[default_dev[0]], "double_sprod");
				kernels[default_dev[0]]["double_sdiv"]=Kernel(*programs[default_dev[0]], "double_sdiv");
				kernels[default_dev[0]]["double_srdiff"]=Kernel(*programs[default_dev[0]], "double_srdiff");
				kernels[default_dev[0]]["double_srdiv"]=Kernel(*programs[default_dev[0]], "double_srdiv");
				kernels[default_dev[0]]["double_sum_second_deriv"]=Kernel(*programs[default_dev[0]], "double_sum_second_deriv");
				kernels[default_dev[0]]["to_rgba"]=Kernel(*programs[default_dev[0]], "to_rgba");

				platform_init[default_dev[0]]=1;
				device_init[default_dev[0]][default_dev[1]]=1;
	
			}

			~_init()
			{
				for(int i=0; i<programs.size(); i++) if(programs[i]) delete programs[i];
			}
		};

		_init _initializer;

		struct event_node {
			event_node **children=NULL, *parent=NULL;
			int size=0;

			Event read, write;
			bool bRead=0, bWrite=0;

			void construct(int n)
			{
				size=n;
				if(children) {
					cerr << "Nodes already initialized\n";
					exit(1);
				}
				if(size) {
					children=new event_node*[size];
					for(int i=0; i<size; i++) children[i]=NULL;
				}
			}

			void destruct()
			{
				if(children) {
					for(int i=0; i<size; i++) if(children[i]) {
						children[i]->destruct();
						delete children[i];
					}
					delete[] children;
				}
			}

			Event *event_pointer(char kind)
			{
				if(kind=='r') {
					bRead=1;
					return &read;
				}
				if(kind=='w') {
					bWrite=1;
					return &write;
				}
				return NULL;
			}

			void fill_children_waitlist(char kind, vector< Event > &waitlist)
			{
				for(int i=0; i<size; i++) if(children[i]) children[i]->fill_children_waitlist(kind, waitlist);
				if((kind=='a' || kind=='r') && bRead) waitlist.push_back(read);
				if((kind=='a' || kind=='w') && bWrite) waitlist.push_back(write);
			}

			vector< Event > gen_waitlist(char kind)
			{
				vector< Event > ret;
				fill_children_waitlist(kind, ret);
				for(event_node* i=parent; i; i=i->parent) {
					if((kind=='a' || kind=='r') && i->bRead) ret.push_back(i->read);
					if((kind=='a' || kind=='w') && i->bWrite) ret.push_back(i->write);
				}
				return ret;
			}

			void push_event(char kind, Event event)
			{
				if(kind=='r') {
					bRead=1;
					read=event;
				} else if(kind=='w') {
					bWrite=1;
					write=event;
				}
			}

			void wait(int kind)
			{
				vector< Event > waitlist=gen_waitlist(kind);
				for(int i=0; i<waitlist.size(); i++) waitlist[i].wait();
			}

			int allocated_children()
			{
				int ret=0;
				for(int i=0; i<size; i++) if(children[i]) ret++;
				return ret;
			}

			void purge()
			{
				for(int i=0; i<size; i++) if(children[i]) {
					event_node *it=children[i];
					it->purge();
					if(!(it->allocated_children() || it->bRead || it->bWrite)) {
						if(it->children) delete[] it->children;
						delete it;
						children[i]=NULL;
					}
				}
				if(bRead && read.getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>()==CL_COMPLETE) bRead=0;
				if(bWrite && write.getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>()==CL_COMPLETE) bWrite=0;
			}

		};

	}

	template<typename T>
	class host_tensor;

	template<typename T>
	class device_tensor
	{
		private:
			device_tensor();
		public:
		friend class host_tensor<T>;
	};
	
	template<>
	class device_tensor<double>
	{
		private:

		int devid[2]={default_dev[0], default_dev[1]};
		cl::Device *device=&devices[devid[0]][devid[1]];
		cl::Context *context=&contexts[devid[0]];
		cl::CommandQueue *queue=&queues[devid[0]][devid[1]];

		cl::Kernel *k_assign=&kernels[devid[0]]["double_assign"];
		cl::Kernel *k_sum=&kernels[devid[0]]["double_sum"];
		cl::Kernel *k_diff=&kernels[devid[0]]["double_diff"];
		cl::Kernel *k_prod=&kernels[devid[0]]["double_prod"];
		cl::Kernel *k_div=&kernels[devid[0]]["double_div"];
		cl::Kernel *k_ssum=&kernels[devid[0]]["double_ssum"];
		cl::Kernel *k_sdiff=&kernels[devid[0]]["double_sdiff"];
		cl::Kernel *k_sprod=&kernels[devid[0]]["double_sprod"];
		cl::Kernel *k_sdiv=&kernels[devid[0]]["double_sdiv"];
		cl::Kernel *k_srdiff=&kernels[devid[0]]["double_srdiff"];
		cl::Kernel *k_srdiv=&kernels[devid[0]]["double_srdiv"];
		cl::Kernel *k_sum_second_deriv=&kernels[devid[0]]["double_sum_second_deriv"];
		cl::Kernel *k_to_rgba=&kernels[devid[0]]["to_rgba"];

		int m_type=0;
		int *m_shape=NULL;
		int m_elems=0;
		cl::Buffer m_data;
		
		int m_data_offset=0; //La puta que te parió OpenCL
		cl::Buffer *m_data_zero=&m_data;

		cl::NDRange m_global;
		cl::NDRange m_local=256;

		device_tensor<double> *m_parent=NULL;

		double m_dx=1;

		event_node *m_events=NULL;

		inline void push(double *data, cl_bool blocking = CL_FALSE )
		{
			std::vector<cl::Event> waitlist=m_events->gen_waitlist('a');
			queue->enqueueWriteBuffer(m_data, blocking, 0, m_elems*sizeof(double), data, &waitlist, m_events->event_pointer('w'));
		}

		inline void pull(double *data, cl_bool blocking = CL_FALSE )
		{
			std::vector<cl::Event> waitlist=m_events->gen_waitlist('w');
			queue->enqueueReadBuffer(m_data, blocking, 0, m_elems*sizeof(double), data, &waitlist, m_events->event_pointer('r'));
		}

		inline void copy_data(const device_tensor<double> &ten)
		{
			std::vector<cl::Event> waitlist=m_events->gen_waitlist('a'), temp=ten.m_events->gen_waitlist('w');
			waitlist.insert(waitlist.end(), temp.begin(), temp.end());
			queue->enqueueCopyBuffer(ten.m_data, m_data, 0, 0, m_elems*sizeof(double), &waitlist, m_events->event_pointer('w'));
			ten.m_events->push_event('r', m_events->write);
		}

		inline void move(device_tensor<double> &ten)
		{
			m_type=ten.m_type;
			m_shape=ten.m_shape;
			m_elems=ten.m_elems;
			m_data=ten.m_data;
			m_global=ten.m_global;
			m_dx=ten.m_dx;
			m_events=ten.m_events;
			
			ten.m_parent=this;
		}

		inline void set_shape(const std::vector<int> shape)
		{
			for(int i=0; i<shape.size(); i++) if(shape[i]<=0) return;
			m_type=shape.size();
			if(m_type) m_shape=new int[m_type];
			m_elems=1;
			for(int i=0; i<m_type; i++) m_elems*=(m_shape[i]=shape[i]);
			m_global=m_elems;
		}

		inline bool same_shape(const device_tensor<double> &A, const device_tensor<double> &B) const
		{
			if(A.m_elems!=B.m_elems) return 0;
			if(A.m_type!=B.m_type) return 0;
			for(int i=0; i<A.m_type; i++) if(A.m_shape[i]!=B.m_shape[i]) return 0;
			return 1;
		}

		inline void alloc()
		{
			if(m_elems) m_data=cl::Buffer(*context, CL_MEM_READ_WRITE, m_elems*sizeof(double));
			m_events=new event_node;
		}

		inline void destruct()
		{
			if(!m_parent) {
				if(m_events) {
					m_events->destruct();
					delete m_events;
				}
				if(m_shape) delete m_shape;
			}
		}

		inline void reset()
		{
			destruct();
			m_elems=0;
			m_type=0;
			m_shape=NULL;
			m_parent=NULL;
			m_data=cl::Buffer();
			m_events=new event_node;
		}

		inline device_tensor<double> subtensor(const int i)
		{
			int elems=m_elems/m_shape[0];
			device_tensor<double> ret;
			ret.m_type=m_type-1;
			ret.m_shape=elems>1?m_shape+1:NULL;
			ret.m_elems=elems;
			ret.m_parent=this;
			ret.m_data_zero=m_data_zero;
			ret.m_data_offset=m_data_offset+i*sizeof(double)*elems;
			ret.m_global=elems;

			if(!m_events->children) {
				m_events->construct(m_shape[0]);
				m_events->children[i]=new event_node;
			} else if(!m_events->children[i]) m_events->children[i]=new event_node;
			ret.m_events=m_events->children[i];
			ret.m_events->parent=m_events;

			if(elems) {
				cl_buffer_region reg={(long unsigned int)ret.m_data_offset, sizeof(double)*elems};
				cl_int err;
				ret.m_data=m_data_zero->createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &reg, &err); //La puta que te parió OpenCL
			}

			return ret;
		}

		public:

		friend class host_tensor<double>;
		operator host_tensor<double>();

		void print_tree(event_node *node)
		{
			if(node) {
				std::cout << "{ (" << node->bRead << ", " << node->bWrite << "), ";
				if(node->children) {
					for(int i=0; i<node->size; i++) {
						std::cout << "[ ";
						if(node->children[i]) print_tree(node->children[i]);
						std::cout << " ]";
					}
				}
				std::cout << " }";
			}
		}

		void print_tree()
		{
			print_tree(m_events);
			std::cout << "\n";
		}

		/*
		Debugging tool
		*/

		//Constructors

		device_tensor<double>()
		{
			//m_events=new event_node;
		};

		device_tensor<double>(const std::vector<int> &shape)
		{
			set_shape(shape);
			alloc();
		}

		device_tensor<double>(const std::vector<int> &shape, const double val)
		{
			set_shape(shape);
			alloc();

			k_assign->setArg(0, m_data);
			k_assign->setArg(1, val);

			queue->enqueueNDRangeKernel(*k_assign, cl::NullRange, m_global, m_local, NULL, m_events->event_pointer('w'));
		}

		device_tensor<double>(const device_tensor<double> &ten)
		{
			m_dx=ten.m_dx;
			set_shape(ten.shape());
			alloc();
			copy_data(ten);
		}

		device_tensor<double>(device_tensor<double> &&ten)
		{
			m_dx=ten.m_dx;
			if(ten.m_parent) {
				set_shape(ten.shape());
				alloc();
				copy_data(ten);
			} else 	move(ten);
		}

		~device_tensor<double>()
		{
			destruct();
		}

		//Operator=

		/*
		device_tensor<double> &operator=(const double scalar)
		{
			if(m_elems!=1) {
				if(m_parent) {
					return device_tensor<double>();
				reset();
				m_elems=1;
				alloc();
			}
			push(&scalar, CL_TRUE);
			return *this;
		}
		*/

		device_tensor<double> &operator=(const device_tensor<double> &ten)
		{
			m_dx=ten.m_dx;
			if(!ten.m_elems) {
				reset();
				return *this;
			}

			if(m_elems!=ten.m_elems || !same_shape(*this, ten)) {
				reset();
				set_shape(ten.shape());
				alloc();
			}
			copy_data(ten);
			return *this;
		}

		device_tensor<double> &operator=(device_tensor<double> &&ten)
		{
			m_dx=ten.m_dx;
			if(!ten.m_elems || m_parent && !same_shape(*this, ten)) {
				reset();
				return *this;
			}

			if(m_parent) copy_data(ten);
			else  {
				reset();
				move(ten);
			}

			return *this;
		}

		device_tensor<double> operator[](const int i)
		{
			if(!m_type) return device_tensor<double>();
			if(i<-m_shape[0] || i>=m_shape[0]) return device_tensor<double>();
			return subtensor(i<0?i+m_shape[0]:i);
		}

		private:

		inline device_tensor<double> ew_bin_op(const device_tensor<double> &A, const device_tensor<double> &B, cl::Kernel *kern) const
		{
			if(!same_shape(A, B)) return device_tensor<double>();

			device_tensor<double> ret(A.shape());

			kern->setArg(0, ret.m_data);
			kern->setArg(1, A.m_data);
			kern->setArg(2, B.m_data);

			std::vector<cl::Event> waitlist=A.m_events->gen_waitlist('w'), temp=B.m_events->gen_waitlist('w');
			waitlist.insert(waitlist.end(), temp.begin(), temp.end());

			queue->enqueueNDRangeKernel(*kern, cl::NullRange, ret.m_global, ret.m_local, &waitlist, ret.m_events->event_pointer('w'));
			A.m_events->push_event('r', ret.m_events->write);
			B.m_events->push_event('r', ret.m_events->write);

			return ret;
		}

		inline device_tensor<double> &ip_ew_bin_op(device_tensor<double> &A, const device_tensor<double> &B, cl::Kernel *kern) const
		{
			if(!same_shape(A, B)) return A=device_tensor<double>();

			kern->setArg(0, A.m_data);
			kern->setArg(1, A.m_data);
			kern->setArg(2, B.m_data);

			std::vector<cl::Event> waitlist=A.m_events->gen_waitlist('a'), temp=B.m_events->gen_waitlist('w');
			waitlist.insert(waitlist.end(), temp.begin(), temp.end());

			queue->enqueueNDRangeKernel(*kern, cl::NullRange, A.m_global, A.m_local, &waitlist, A.m_events->event_pointer('w'));
			B.m_events->push_event('r', A.m_events->write);

			return A;
		}

		inline device_tensor<double> ew_bin_sop(const device_tensor<double> &A, const double k, cl::Kernel *kern) const
		{
			device_tensor<double> ret(A.shape());

			kern->setArg(0, ret.m_data);
			kern->setArg(1, A.m_data);
			kern->setArg(2, k);

			std::vector<cl::Event> waitlist=A.m_events->gen_waitlist('w');

			queue->enqueueNDRangeKernel(*kern, cl::NullRange, ret.m_global, ret.m_local, &waitlist, ret.m_events->event_pointer('w'));
			A.m_events->push_event('r', ret.m_events->write);

			return ret;
		}

		inline device_tensor<double> ip_ew_bin_sop(device_tensor<double> &A, const double k, cl::Kernel *kern) const
		{
			kern->setArg(0, A.m_data);
			kern->setArg(1, A.m_data);
			kern->setArg(2, k);

			std::vector<cl::Event> waitlist=A.m_events->gen_waitlist('w');

			queue->enqueueNDRangeKernel(*kern, cl::NullRange, A.m_global, A.m_local, &waitlist, A.m_events->event_pointer('w'));

			return A;
		}

		public:

		//Tensor-tensor operations

		device_tensor<double> operator+(const device_tensor<double> &ten) const
		{
			return ew_bin_op(*this, ten, k_sum);
		}

		device_tensor<double> operator+=(const device_tensor<double> &ten)
		{
			return ip_ew_bin_op(*this, ten, k_sum);
		}

		device_tensor<double> operator-(const device_tensor<double> &ten) const
		{
			return ew_bin_op(*this, ten, k_diff);
		}

		device_tensor<double> operator-=(const device_tensor<double> &ten)
		{
			return ip_ew_bin_op(*this, ten, k_diff);
		}

		device_tensor<double> operator*(const device_tensor<double> &ten) const
		{
			return ew_bin_op(*this, ten, k_prod);
		}

		device_tensor<double> operator*=(const device_tensor<double> &ten)
		{
			return ip_ew_bin_op(*this, ten, k_prod);
		}

		device_tensor<double> operator/(const device_tensor<double> &ten) const
		{
			return ew_bin_op(*this, ten, k_div);
		}

		device_tensor<double> operator/=(const device_tensor<double> &ten)
		{
			return ip_ew_bin_op(*this, ten, k_div);
		}

		//Tensor-scalar operators

		device_tensor<double> operator+(const double k) const
		{
			return ew_bin_sop(*this, k, k_ssum);
		}

		friend device_tensor<double> operator+(const double k, const device_tensor<double> ten)
		{
			return ten.ew_bin_sop(ten, k, ten.k_ssum);
		}

		device_tensor<double> operator+=(const double k)
		{
			return ip_ew_bin_sop(*this, k, k_ssum);
		}

		device_tensor<double> operator-(const double k) const
		{
			return ew_bin_sop(*this, k, k_sdiff);
		}

		friend device_tensor<double> operator-(const double k, const device_tensor<double> ten)
		{
			return ten.ew_bin_sop(ten, k, ten.k_srdiff);
		}

		device_tensor<double> operator-=(const double k)
		{
			return ip_ew_bin_sop(*this, k, k_sdiff);
		}

		device_tensor<double> operator*(const double k) const
		{
			return ew_bin_sop(*this, k, k_sprod);
		}

		friend device_tensor<double> operator*(const double k, const device_tensor<double> ten)
		{
			return ten.ew_bin_sop(ten, k, ten.k_sprod);
		}

		device_tensor<double> operator*=(const double k)
		{
			return ip_ew_bin_sop(*this, k, k_sprod);
		}

		device_tensor<double> operator/(const double k) const
		{
			return ew_bin_sop(*this, k, k_sdiv);
		}

		friend device_tensor<double> operator/(const double k, const device_tensor<double> ten)
		{
			return ten.ew_bin_sop(ten, k, ten.k_srdiv);
		}

		device_tensor<double> operator/=(const double k)
		{
			return ip_ew_bin_sop(*this, k, k_sdiv);
		}

		friend device_tensor<double> laplacian(const device_tensor<double> &ten)
		{
			device_tensor<double> ret(ten.shape(), 0);
			int step=ten.m_elems;

			if(ten.m_events->bWrite) ret.m_events->push_event('w', ten.m_events->write);
			for(int i=0; i<ten.m_type; i++) {
				ten.k_sum_second_deriv->setArg(0, ret.m_data);
				ten.k_sum_second_deriv->setArg(1, ten.m_data);
				ten.k_sum_second_deriv->setArg(2, step/=ten.m_shape[i]);
				ten.k_sum_second_deriv->setArg(3, ten.m_shape[i]);

				std::vector<cl::Event> waitlist=ret.m_events->gen_waitlist('w');
				ten.queue->enqueueNDRangeKernel(*ten.k_sum_second_deriv, cl::NullRange, ret.m_global, ret.m_local, &waitlist, ret.m_events->event_pointer('w'));
			}

			std::vector<cl::Event> waitlist=ret.m_events->gen_waitlist('w');
			ten.k_sprod->setArg(0, ret.m_data);
			ten.k_sprod->setArg(1, ret.m_data);
			ten.k_sprod->setArg(2, 1/ten.m_dx/ten.m_dx);
			ten.queue->enqueueNDRangeKernel(*ten.k_sprod, cl::NullRange, ret.m_global, ret.m_local, &waitlist, ret.m_events->event_pointer('w'));

			ten.m_events->push_event('r', ret.m_events->write);
			ret.m_dx=ten.m_dx;

			return ret;
		}

		//Host info facilities

		template<typename T>
		friend std::ostream& operator<<(std::ostream &out, device_tensor<T> ten);

		std::vector<int> shape() const
		{
			return std::vector<int>(m_shape, m_shape+m_type);
		}

		int elements() const
		{
			return m_elems;
		}

		device_tensor<double> reshape(const std::vector<int> &shape) const
		{
			int elems=1;
			for(int i=0; i<shape.size(); i++)
				if(shape[i]<=0) return device_tensor<double>();
				else elems*=shape[i];
			if(elems!=m_elems) return device_tensor<double>();
			device_tensor<double> ret(shape);
			ret.copy_data(*this);
			return ret;
		}

		double get_dx()
		{
			return m_dx;
		}

		void set_dx(double dx)
		{
			m_dx=dx;
		}

		//Event handling tools

		void flush(char kind='a')
		{
			m_events->wait(kind);
			if(kind=='a' || kind=='r') m_events->bRead=0;
			if(kind=='a' || kind=='w') m_events->bWrite=0;
			if(kind=='a') {
				m_events->destruct();
				m_events->children=NULL;
			} else m_events->purge();
			m_events->size=0;
		}

		cl::Buffer RGBA_buffer(const double min, const double max)
		{
			if(m_type!=2) return cl::Buffer();
			cl::Buffer ret(*context, CL_MEM_READ_WRITE, 4*m_elems*sizeof(float));

			k_to_rgba->setArg(0, ret);
			k_to_rgba->setArg(1, m_data);
			k_to_rgba->setArg(2, min);
			k_to_rgba->setArg(3, max);

			std::vector<cl::Event> waitlist=m_events->gen_waitlist('w');
			queue->enqueueNDRangeKernel(*k_to_rgba, cl::NullRange, m_global, m_local, &waitlist, m_events->event_pointer('w'));
			return ret;
		}

	};

	template<typename K>
	class host_tensor
	{
		private:

		int m_type=0;
		int *m_shape=NULL;
		int m_elems=0;
		K *m_data;
		
		host_tensor<K> *m_parent=NULL;

		double m_dx=1;

		event_node *m_events=NULL;

		inline void copy_data(const host_tensor<K> &ten)
		{
			ten.m_events->wait('w');
			m_events->wait('a');
			for(int i=0; i<m_elems; i++)
				m_data[i]=ten.m_data[i];
		}

		inline void move(host_tensor<K> &ten)
		{
			m_type=ten.m_type;
			m_shape=ten.m_shape;
			m_elems=ten.m_elems;
			m_data=ten.m_data;
			m_dx=ten.m_dx;
			m_events=ten.m_events;
			
			ten.m_parent=this;
		}

		inline void set_shape(const std::vector<int> shape)
		{
			for(int i=0; i<shape.size(); i++) if(shape[i]<=0) return;
			m_type=shape.size();
			if(m_type) m_shape=new int[m_type];
			m_elems=1;
			for(int i=0; i<m_type; i++) m_elems*=(m_shape[i]=shape[i]);
		}

		inline bool same_shape(const host_tensor<K> &A, const host_tensor<K> &B) const
		{
			if(A.m_elems!=B.m_elems) return 0;
			if(A.m_type!=B.m_type) return 0;
			for(int i=0; i<A.m_type; i++) if(A.m_shape[i]!=B.m_shape[i]) return 0;
			return 1;
		}

		inline void alloc()
		{
			if(m_elems) m_data=new K[m_elems];
			m_events=new event_node;
		}

		inline void destruct()
		{
			if(!m_parent) {
				if(m_data) {
					m_events->wait('a');
					delete[] m_data;
				}
				if(m_events) {
					m_events->destruct();
					delete m_events;
				}
				if(m_shape) delete m_shape;
			}
		}

		inline void reset()
		{
			destruct();
			m_elems=0;
			m_type=0;
			m_shape=NULL;
			m_parent=NULL;
			m_data=NULL;
			m_events=new event_node;
		}

		inline host_tensor<K> subtensor(const int i)
		{
			int elems=m_elems/m_shape[0];
			host_tensor<K> ret;
			ret.m_type=m_type-1;
			ret.m_shape=elems>1?m_shape+1:NULL;
			ret.m_elems=elems;
			ret.m_parent=this;
			if(m_data) ret.m_data=m_data+i*elems;

			if(!m_events->children) {
				m_events->construct(m_shape[0]);
				m_events->children[i]=new event_node;
			} else if(!m_events->children[i]) m_events->children[i]=new event_node;
			ret.m_events=m_events->children[i];
			ret.m_events->parent=m_events;

			return ret;
		}

		public:

		/*
		Debugging tool

		void print_tree(event_node *node)
		{
			if(node) {
				std::cout << "{ (" << node->bRead << ", " << node->bWrite << "), ";
				if(node->children) {
					for(int i=0; i<node->size; i++) {
						std::cout << "[ ";
						if(node->children[i]) print_tree(node->children[i]);
						std::cout << " ]";
					}
				}
				std::cout << " }";
			}
		}

		void print_tree()
		{
			print_tree(m_events);
			std::cout << "\n";
		}

		 */

		//Constructors

		friend class device_tensor<K>;
		operator device_tensor<K>();

		host_tensor<K>(){};

		host_tensor<K>(const std::vector<int> &shape)
		{
			set_shape(shape);
			alloc();
		}

		host_tensor<K>(const std::vector<int> &shape, const K val)
		{
			set_shape(shape);
			alloc();

			for(int i=0; i<m_elems; i++) m_data[i]=val;
		}

		host_tensor<K>(const host_tensor<K> &ten)
		{
			m_dx=ten.m_dx;
			set_shape(ten.shape());
			alloc();
			copy_data(ten);
		}

		host_tensor<K>(host_tensor<K> &&ten)
		{
			m_dx=ten.m_dx;
			if(ten.m_parent) {
				set_shape(ten.shape());
				alloc();
				copy_data(ten);
			} else 	move(ten);
		}

		~host_tensor<K>()
		{
			destruct();
		}

		//Operator=

		host_tensor<K> &operator=(const K scalar)
		{
			if(m_elems!=1) {
				if(m_parent)
					return *this=host_tensor<K>();
				reset();
				m_elems=1;
				alloc();
			}
			m_events->wait('a');
			*m_data=scalar;
			return *this;
		}

		host_tensor<K> &operator=(const host_tensor<K> &ten)
		{
			m_dx=ten.m_dx;
			if(!ten.m_elems) {
				reset();
				return *this;
			}

			if(m_elems!=ten.m_elems || !same_shape(*this, ten)) {
				reset();
				set_shape(ten.shape());
				alloc();
			}
			copy_data(ten);
			return *this;
		}

		host_tensor<K> &operator=(host_tensor<K> &&ten)
		{
			m_dx=ten.m_dx;
			if(!ten.m_elems || m_parent && !same_shape(*this, ten)) {
				reset();
				return *this;
			}

			if(m_parent) copy_data(ten);
			else  {
				reset();
				move(ten);
			}

			return *this;
		}

		host_tensor<K> operator[](const int i)
		{
			if(!m_type) return host_tensor<K>();
			if(i<-m_shape[0] || i>=m_shape[0]) return host_tensor<K>();
			return subtensor(i<0?i+m_shape[0]:i);
		}

		private:

		inline host_tensor<K> ew_bin_op(const host_tensor<K> &A, const host_tensor<K> &B, std::function<K(K, K)> f) const
		{
			if(!same_shape(A, B)) return host_tensor<K>();

			host_tensor<K> ret(A.shape());
			A.m_events->wait('w');
			B.m_events->wait('w');
			for(int i=0; i<m_elems; i++) ret.m_data[i]=f(A.m_data[i], B.m_data[i]);

			return ret;
		}

		inline host_tensor<K> &ip_ew_bin_op(host_tensor<K> &A, const host_tensor<K> &B, std::function<K(K, K)> f) const
		{
			if(!same_shape(A, B)) return A=host_tensor<K>();

			A.m_events->wait('a');
			B.m_events->wait('w');
			for(int i=0; i<m_elems; i++) A.m_data[i]=f(A.m_data[i], B.m_data[i]);

			return A;
		}

		inline host_tensor<K> ew_bin_sop(const host_tensor<K> &A, const K k, std::function<K(K, K)> f) const
		{
			host_tensor<K> ret(A.shape());

			A.m_events->wait('w');
			for(int i=0; i<m_elems; i++) ret.m_data[i]=f(A.m_data[i], k);

			return ret;
		}

		inline host_tensor<K> ip_ew_bin_sop(host_tensor<K> &A, const K k, std::function<K(K, K)> f) const
		{
			A.m_events->wait('a');
			for(int i=0; i<m_elems; i++) A.m_data[i]=f(A.m_data[i], k);
			return A;
		}

		public:

		//Tensor-tensor operations

		host_tensor<K> operator+(const host_tensor<K> &ten) const
		{
			return ew_bin_op(*this, ten, [](K x, K y)->K{ return x+y; });
		}
	
		host_tensor<K> operator+=(const host_tensor<K> &ten)
		{
			return ip_ew_bin_op(*this, ten, [](K x, K y)->K{ return x+y; });
		}

		host_tensor<K> operator-(const host_tensor<K> &ten) const
		{
			return ew_bin_op(*this, ten, [](K x, K y)->K{ return x-y; });
		}

		host_tensor<K> operator-=(const host_tensor<K> &ten)
		{
			return ip_ew_bin_op(*this, ten, [](K x, K y)->K{ return x-y; });
		}

		host_tensor<K> operator*(const host_tensor<K> &ten) const
		{
			return ew_bin_op(*this, ten, [](K x, K y)->K{ return x*y; });
		}

		host_tensor<K> operator*=(const host_tensor<K> &ten)
		{
			return ip_ew_bin_op(*this, ten, [](K x, K y)->K{ return x*y; });
		}

		host_tensor<K> operator/(const host_tensor<K> &ten) const
		{
			return ew_bin_op(*this, ten, [](K x, K y)->K{ return x/y; });
		}

		host_tensor<K> operator/=(const host_tensor<K> &ten)
		{
			return ip_ew_bin_op(*this, ten, [](K x, K y)->K{ return x/y; });
		}

		//Tensor-scalar operators

		host_tensor<K> operator+(const K k) const
		{
			return ew_bin_sop(*this, k, [](K x, K y)->K{ return x+y; });
		}

		friend host_tensor<K> operator+(const K k, const host_tensor<K> ten)
		{
			return ten.ew_bin_sop(ten, k, [](K x, K y)->K{ return x+y; });
		}

		host_tensor<K> operator+=(const K k)
		{
			return ip_ew_bin_sop(*this, k, [](K x, K y)->K{ return x+y; });
		}

		host_tensor<K> operator-(const K k) const
		{
			return ew_bin_sop(*this, k, [](K x, K y)->K{ return x-y; });
		}

		friend host_tensor<K> operator-(const K k, const host_tensor<K> ten)
		{
			return ten.ew_bin_sop(ten, k, [](K x, K y)->K{ return y-x; });
		}

		host_tensor<K> operator-=(const K k)
		{
			return ip_ew_bin_sop(*this, k, [](K x, K y)->K{ return x-y; });
		}

		host_tensor<K> operator*(const K k) const
		{
			return ew_bin_sop(*this, k, [](K x, K y)->K{ return x*y; });
		}

		friend host_tensor<K> operator*(const K k, const host_tensor<K> ten)
		{
			return ten.ew_bin_sop(ten, k, [](K x, K y)->K{ return x*y; });
		}

		host_tensor<K> operator*=(const K k)
		{
			return ip_ew_bin_sop(*this, k, [](K x, K y)->K{ return x*y; });
		}

		host_tensor<K> operator/(const K k) const
		{
			return ew_bin_sop(*this, k, [](K x, K y)->K{ return x/y; });
		}

		friend host_tensor<K> operator/(const K k, const host_tensor<K> ten)
		{
			return ten.ew_bin_sop(ten, k, [](K x, K y)->K{ return y/x; });
		}

		host_tensor<K> operator/=(const K k)
		{
			return ip_ew_bin_sop(*this, k, [](K x, K y)->K{ return x/y; });
		}

		/*
		friend host_tensor<K> laplacian(const host_tensor<double> &ten)
		{
			host_tensor<double> ret(ten.shape(), 0);
			int step=ten.m_elems;

			if(ten.m_events->bWrite) ret.m_events->push_event('w', ten.m_events->write);
			for(int i=0; i<ten.m_type; i++) {
				ten.k_sum_second_deriv->setArg(0, ret.m_data);
				ten.k_sum_second_deriv->setArg(1, ten.m_data);
				ten.k_sum_second_deriv->setArg(2, step/=ten.m_shape[i]);
				ten.k_sum_second_deriv->setArg(3, ten.m_shape[i]);

				std::vector<cl::Event> waitlist=ret.m_events->gen_waitlist('w');
				ten.queue->enqueueNDRangeKernel(*ten.k_sum_second_deriv, cl::NullRange, ret.m_global, ret.m_local, &waitlist, ret.m_events->event_pointer('w'));
			}

			std::vector<cl::Event> waitlist=ret.m_events->gen_waitlist('w');
			ten.k_sprod->setArg(0, ret.m_data);
			ten.k_sprod->setArg(1, ret.m_data);
			ten.k_sprod->setArg(2, 1/ten.m_dx/ten.m_dx);
			ten.queue->enqueueNDRangeKernel(*ten.k_sprod, cl::NullRange, ret.m_global, ret.m_local, &waitlist, ret.m_events->event_pointer('w'));

			ret.m_dx=ten.m_dx;

			return ret;
		}
		*/

		//Host info facilities

		private:
		std::ostream &naive_output(std::ostream &out, host_tensor<K> ten)
		{
			if(ten.m_type==0) {
				if(ten.m_elems) out << *ten.m_data;
			} else {
				out << "[ ";
				for(int i=0; i<ten.m_shape[0]; i++) naive_output(out, ten.subtensor(i)) << (i+1==ten.m_shape[0]?" ":", ");
				out << "]";
			}
		}


		public:

		template<typename T>
		friend std::ostream& operator<<(std::ostream &out, host_tensor<T> ten);

		std::vector<int> shape() const
		{
			return std::vector<int>(m_shape, m_shape+m_type);
		}

		int elements() const
		{
			return m_elems;
		}

		host_tensor<double> reshape(const std::vector<int> &shape) const
		{
			int elems=1;
			for(int i=0; i<shape.size(); i++)
				if(shape[i]<=0) return host_tensor<double>();
				else elems*=shape[i];
			if(elems!=m_elems) return host_tensor<double>();
			host_tensor<double> ret(shape);
			ret.copy_data(*this);
			return ret;
		}

		double get_dx()
		{
			return m_dx;
		}

		void set_dx(double dx)
		{
			m_dx=dx;
		}

		//Event handling tools

		void flush(char kind='a')
		{
			m_events->wait(kind);
			if(kind=='a') {
				m_events->destruct();
				m_events->children=NULL;
			} else m_events->purge();
		}

	};

	device_tensor<double>::operator host_tensor<double>()
	{
		host_tensor<double> ret(shape());
		ret.m_dx=m_dx;

		std::vector<cl::Event> waitlist=m_events->gen_waitlist('w');
		queue->enqueueReadBuffer(m_data, CL_FALSE, 0, m_elems*sizeof(double), ret.m_data, &waitlist, m_events->event_pointer('r'));
		ret.m_events->push_event('w', m_events->read);
		return ret;
	}

	template<typename T>
	host_tensor<T>::operator device_tensor<T>()
	{
		device_tensor<double> ret(shape());
		ret.m_dx=m_dx;

		std::vector<cl::Event> waitlist=m_events->gen_waitlist('w');
		ret.queue->enqueueWriteBuffer(ret.m_data, CL_FALSE, 0, m_elems*sizeof(double), m_data, &waitlist, m_events->event_pointer('r'));
		ret.m_events->push_event('w', m_events->read);
		return ret;
	}

	template<typename K>
	std::ostream& operator<<(std::ostream &out, device_tensor<K> ten)
	{
		return out << (host_tensor<K>)ten;
	}

	template<typename K>
	std::ostream& operator<<(std::ostream &out, host_tensor<K> ten)
	{
		ten.m_events->wait('w');
		return ten.naive_output(out, ten);
	}

}

//WAVE THINGIES

void output_matrix(std::ostream &out, bru::device_tensor<double> d_mat)
{
	int I=d_mat.shape()[0];
	int J=d_mat.shape()[1];
	bru::host_tensor<double> h_mat=d_mat;
	for(int i=0; i<I; i++) for(int j=0; j<J; j++) out << h_mat[i][j] << (j==J-1?"\n":"\t");
}

template<typename T>
void rk4_step(T &y, double t, double dt, std::function<T(T, double)> f)
{
	T y_0=y, k;

	k=dt/2*f(y_0, t); //		k1 / 2
	y+=(1.0/3)*k;
	k=dt*f(y_0+k, t+dt/2); //	k2
	y+=(1.0/3)*k;
	k=dt*f(y_0+.5*k, t+dt/2); //	k3
	y+=(1.0/3)*k;
	k=dt*f(y+k, t+dt);
	y+=(1.0/3)*k;
	y.flush();
}

std::function<bru::device_tensor<double>(bru::device_tensor<double>, double)> wave_eq(bru::device_tensor<double> mask, double speed=1)
{
	return [mask, speed] (bru::device_tensor<double> phi, double t)->bru::device_tensor<double> {
		bru::device_tensor<double> ret(phi.shape());
		ret[0]=phi[1]; //TODO: WHY T.F. doesn't the in-place operation work
		ret[0]*=mask;
		ret[1]=speed*speed*laplacian(phi[0]);
		return ret;
	};
}

int main()
{
	using namespace bru;
	using namespace std;

	int len=101;
	int size=len*len;
	double dx=1.0/len, dt=dx;

	device_tensor<double> phi({2, len, len}, 0);
	phi.set_dx(dx);

	//Field's original condition
	host_tensor<double> phi0=phi[0];
	double radius=0.5;
	auto f=[](double x)->double{  return x*x>=0.25?0:exp(4+1/(x-0.5)-1/(x+0.5)); };
	for(int i=0; i<len; i++) for(int j=0; j<len; j++) phi0[i][j]=f(sqrt(((i+0.5)*dx-0.5)*((i+0.5)*dx-0.5)+((j+0.5)*dx-0.5)*((j+0.5)*dx-0.5))/radius);
	cout << phi0 << endl;
	phi[0]=phi0;

	/*
	device_tensor<double> mask({len, len}, 1);
	mask[0]=mask[-1]=device_tensor<double>(vector<int>{len}, 0);
	mask[1][0]=mask[1][-1]=device_tensor<double>(vector<int>{}, 0);
	for(int i=2; i<len-1; i++) mask[i]=mask[1];
	*/

	host_tensor<double> mask=device_tensor<double>({len, len}, 1);
	for(int i=0; i<len; i++) for(int j=0; j<len; j++) if(((i+0.5)*dx-0.5)*((i+0.5)*dx-0.5)+((j+0.5)*dx-0.5)*((j+0.5)*dx-0.5)>=0.23) mask[i][j]=0;

	auto waveq=wave_eq(mask);
	std::ofstream outf;
	int spf=100/(101*dx); //steps per frame 
	int frames=100;
	std::cout << "Frames: " << frames << "\tSteps per frame: " << spf << "\n";
	for(int i=0; i<frames*spf; i++) {
		//TODO: use for loop for generating frames instead of outf
		//device_tensor<double>::to_rgba(-1, 1) should return the buffer in float RGBA for each frame
		//call rk4_step.flush() before plotting to make sure the buffer has finished generating
		if(i%spf==0) {
			outf.open(std::to_string(i/spf));
			output_matrix(outf, phi[0]);
			outf.close();
		}
		rk4_step(phi, i*dt, dt, waveq);
	}

	return 0;
}
