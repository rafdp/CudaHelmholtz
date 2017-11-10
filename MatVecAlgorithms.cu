

struct ModifyKMatrix
{
    point_t source;
    const complex_t uiCoeff;
    const float w2h3;
__host__
    ModifyKMatrix (point_t source_, 
                   const complex_t uiCoeff_, 
		   const float w2h3_) : 
        source  (source_),
	uiCoeff (uiCoeff_),
	w2h3    (w2h3_)
    {}
	
__device__
    complex_t operator() (complex_t& k, point_t& pos)
    {
        
        point_t dr = {source.x - pos.x,
                      source.y - pos.y,
                      source.z - pos.z};
        float len = dr.len ();
        complex_t result = w2h3 * thrust::exp (uiCoeff * len) / (4 * 3.141592f * len) * k;
        return result;
    }
};

struct QLReduction
{
    const point_t receiver;
    complex_t * deviceLambdaPtr;
    point_t * deviceIndexesPtr;
    complex_t * deviceKMatrixPtr;
    const pointInt_t discreteBlockSize;
    const complex_t uiCoeff;

    __host__
	QLReduction (point_t receiver_,
                     complex_t * deviceLambdaPtr_,
                     point_t * deviceIndexesPtr_,
                     complex_t * deviceKMatrixPtr_,
		     const pointInt_t discreteBlockSize_,
		     const complex_t uiCoeff_) :
        receiver          (receiver_),
        deviceLambdaPtr   (deviceLambdaPtr_),
        deviceIndexesPtr  (deviceIndexesPtr_),
        deviceKMatrixPtr  (deviceKMatrixPtr_),
	discreteBlockSize (discreteBlockSize_),
	uiCoeff           (uiCoeff_)
    {}

    __device__
	complex_t operator()(int idx) const
	{
		point_t& r = *(deviceIndexesPtr + idx);

		point_t dr = {r.x - receiver.x +
                                      discreteBlockSize.x / 2.0,
                              r.y - receiver.y +
                                      discreteBlockSize.y / 2.0,
                              r.z - receiver.z +
                                      discreteBlockSize.z / 2.0};

		float len = dr.len ();

        	if (len < 0.0000001 && len > 0.0000001) 
		     return complex_t (0.0f, 0.0f);
		return (*(deviceKMatrixPtr + idx)) * (complex_t (1.0f, 0.0f) + *(deviceLambdaPtr + idx)) * 
                           thrust::exp(uiCoeff * len) / (4 * 3.141592f * len);
	}
};

struct IndexFromSequence
{
    const pointInt_t discretizationSize;
    const pointInt_t discreteBlockSize;
    const point_t anomalyPos;

    __host__
    IndexFromSequence (const pointInt_t discretizationSize_,
		       const pointInt_t discreteBlockSize_,
		       const point_t    anomalyPos_) : 
        discretizationSize (discretizationSize_),
	discreteBlockSize  (discreteBlockSize_),
	anomalyPos         (anomalyPos_)
    {}

    __device__
    point_t operator() (int idx) const
    {
        const pointInt_t& ds = discretizationSize;
        point_t point = { 1.0f * (idx % ds.x),
                          1.0f * ((idx / ds.x) % ds.y),
                          1.0f * (idx / (ds.x*ds.y))};
        point = {(float) ((point.x + 0.5f)*discreteBlockSize.x + anomalyPos.x),
                 (float) ((point.y + 0.5f)*discreteBlockSize.y + anomalyPos.y),
                 (float) ((point.z + 0.5f)*discreteBlockSize.z + anomalyPos.z)};
        return point;
    }
};

#ifdef FFT_FUNCTOR

#define CENTER_INDEX (2*size.y+1)*size.x
struct FillRadialQ_lq
{
    point_t * deviceIndexesPtr;
    complex_t* Q_lq;
    pointInt_t size;
    int l;
    int q;
    const float w2h3;
    const complex_t uiCoeff;
    
    __host__
    FillRadialQ_lq (point_t * deviceIndexesPtr_,
                    complex_t * Q_lq_,
                    pointInt_t size_,
                    int l_,
                    int q_,
		    const float w2h3_,
		    const complex_t uiCoeff_) :
        deviceIndexesPtr (deviceIndexesPtr_),
        Q_lq             (Q_lq_),
        size             (size_),
        l                (l_),
        q                (q_),
        w2h3             (w2h3_),
        uiCoeff          (uiCoeff_)
    {}
    
    __device__
    void operator()(int idx) const
    {
	if (idx < 2*size.x) *(Q_lq + idx) = complex_t (0.0f, 0.0f);
	if (idx < 2*size.y) *(Q_lq + idx*2*size.x) = complex_t (0.0f, 0.0f);
	
        int idxy = idx % size.x;
        int idxx = idx / size.x;
        
        int emIdx = l*size.x*size.y;
        int recIdx = q*size.x*size.y + idxy * size.x + idxx;
        
       	//printf ("!!!!!!!!!!Q_lq fill, got idx %d, x %d, y %d, em %d, rec %d\n", idx, idxx, idxy, emIdx, recIdx);
        point_t em = *(deviceIndexesPtr + emIdx);
        if (emIdx == recIdx)
        {
            *(Q_lq + CENTER_INDEX)  = complex_t (0.0f, 0.0f); 
	    return;
        }
        point_t rec = *(deviceIndexesPtr + recIdx);
        
        point_t dr = {rec.x - em.x,
            rec.y - em.y,
            rec.z - em.z};
        
        float len = dr.len ();
	complex_t fill_val = (w2h3 * thrust::exp (uiCoeff * len) / (4 * (3.141592f) * len));    
        *(Q_lq + CENTER_INDEX + (size.x*2)*idxy + idxx) =
        *(Q_lq + CENTER_INDEX - (size.x*2)*idxy + idxx) =
        *(Q_lq + CENTER_INDEX + (size.x*2)*idxy - idxx) =
        *(Q_lq + CENTER_INDEX - (size.x*2)*idxy - idxx) = fill_val;
        
    }
        
};


struct FillRadialQ
{
    point_t * deviceIndexesPtr;
    complex_t* Q_full;
    pointInt_t size;
    const float w2h3;
    const complex_t uiCoeff;
    
    __host__
    FillRadialQ (point_t * deviceIndexesPtr_,
                 complex_t * Q_full_,
                 pointInt_t size_,
		 const float w2h3_,
		 const complex_t uiCoeff_) :
        deviceIndexesPtr (deviceIndexesPtr_),
        Q_full           (Q_full_),
        size             (size_),
        w2h3             (w2h3_),
        uiCoeff          (uiCoeff_)
    {}
    
    __device__
    void operator()(int idx) const
    {
	int idxx = idx % size.x;
	idx /= size.x;
	int idxy = idx % size.y;
	idx /= size.y;
	int q = idx % size.z;
	int l = idx / size.z;

        int gridSize = size.x*size.y*4;
	if (idxy == 0) *(Q_full + (l*size.z + q) * gridSize + idxx) = complex_t (0.0f, 0.0f);
	if (idxx == 0) *(Q_full + (l*size.z + q) * gridSize + idxy * 2*size.x) = complex_t (0.0f, 0.0f);
        
        int emIdx = l*size.x*size.y;
        int recIdx = q*size.x*size.y + idxy * size.x + idxx;
        
        if (emIdx == recIdx)
        {
            *(Q_full + (l*size.z + q) * gridSize + CENTER_INDEX)  = complex_t (0.0f, 0.0f); 
	    return;
        }
        point_t em = *(deviceIndexesPtr + emIdx);
        point_t rec = *(deviceIndexesPtr + recIdx);
        
        point_t dr = {rec.x - em.x,
                      rec.y - em.y,
                      rec.z - em.z};
        
        float len = dr.len ();
	complex_t fill_val = (w2h3 * thrust::exp (uiCoeff * len) / (4 * (3.141592f) * len));    
        *(Q_full + (l*size.z + q) * gridSize + CENTER_INDEX + (size.x*2)*idxy + idxx) =
        *(Q_full + (l*size.z + q) * gridSize + CENTER_INDEX - (size.x*2)*idxy + idxx) =
        *(Q_full + (l*size.z + q) * gridSize + CENTER_INDEX + (size.x*2)*idxy - idxx) =
        *(Q_full + (l*size.z + q) * gridSize + CENTER_INDEX - (size.x*2)*idxy - idxx) = fill_val;
        
    }
        
};


struct FillV
{
    point_t * deviceIndexesPtr;
    complex_t* source;
    complex_t* V;
    pointInt_t size;
    const point_t sourcePos;
    const complex_t uiCoeff;
    
    __host__
    FillV (point_t * deviceIndexesPtr_,
	   complex_t * source_,
           complex_t * V_,
           pointInt_t size_,
	   const point_t sourcePos_,
	   const complex_t uiCoeff_) :
        deviceIndexesPtr (deviceIndexesPtr_),
        source           (source_),
        V                (V_),
        size             (size_),
	sourcePos        (sourcePos_),
	uiCoeff          (uiCoeff_)
    {}

    __device__
    void operator()(int idx) 
    {
        int idxOld = idx;
        int idxx = idx % size.x;
	idx /= size.x;
        int idxy = idx % size.y;
	int idxz = idx / size.y;
        idx *= size.x; 
        int recIdx = idxz*size.x*size.y + idxy * size.x + idxx;
	
	point_t rec = *(deviceIndexesPtr + recIdx);
        point_t dr = {sourcePos.x - rec.x,
                      sourcePos.y - rec.y,
     		      sourcePos.z - rec.z};

        float len = dr.len ();
	complex_t value = thrust::exp (uiCoeff * len) / (4 * 3.141592f * len); 
	*(V + idxz*4*size.x*size.y + CENTER_INDEX + (size.x*2)*idxy + idxx) =
            *(source + idxOld) * value;
	
	if (idxy)         *(V + idxz*4*size.x*size.y + CENTER_INDEX - (size.x*2)*idxy + idxx) = complex_t (0.0f, 0.0f);
	if (idxx)         *(V + idxz*4*size.x*size.y + CENTER_INDEX + (size.x*2)*idxy - idxx) = complex_t (0.0f, 0.0f);
	if (idxx && idxy) *(V + idxz*4*size.x*size.y + CENTER_INDEX - (size.x*2)*idxy - idxx) = complex_t (0.0f, 0.0f);
        if (!idxy)
	{
	          *(V + idxz*4*size.x*size.y + idxx) = complex_t (0.0f, 0.0f);
	          *(V + idxz*4*size.x*size.y + idxx + size.x) = complex_t (0.0f, 0.0f);
	          *(V + idxz*4*size.x*size.y + idxx*size.x*2 + size.x*2*size.y) = complex_t (0.0f, 0.0f);
        if (idxx) *(V + idxz*4*size.x*size.y + idxx*size.x*2) = complex_t (0.0f, 0.0f);
	}
    }
};


const int OP_ADD = '+',
          OP_SUB = '-',
	  OP_MUL = '*',
	  OP_SET = '=';

struct ElementwiseOperation
{
    complex_t* modifiable;
    complex_t* source;
    int op;
    __host__
    ElementwiseOperation (complex_t * mod_,
                           complex_t * source_,
			   int op_) :
        modifiable (mod_),
        source     (source_),
	op         (op_)
    {}

    __device__
    void operator()(int idx) const
    {
        if (op == OP_ADD) *(modifiable + idx) += *(source + idx);
	else
        if (op == OP_SUB) *(modifiable + idx) -= *(source + idx);
	else
        if (op == OP_MUL) *(modifiable + idx) *= *(source + idx);
	else 
	if (op == OP_SET) *(modifiable + idx)  = *(source + idx);
    }
};

struct BlockElementwiseOperation
{
    complex_t* modifiable;
    complex_t* source;
    int op;
    pointInt_t size;
    __host__
    BlockElementwiseOperation (complex_t * mod_,
                           complex_t * source_,
			   int op_,
			   pointInt_t size_) :
        modifiable (mod_),
        source     (source_),
	op         (op_),
	size       (size_)
    {}

    __device__
    void operator()(int idx) const
    {
	int idxx = idx % (2 * size.x);
	idx /= 2*size.x;
	int idxy = idx % (2 * size.y);
        int idxz = idx / (2 * size.y);
        int modIdx = idxz * 4 * size.x * size.y + idxy*2*size.x + idxx;
	int srcIdx = (idxz*size.z)*4*size.x*size.y + idxy * 2*size.x + idxx;

        if (op == OP_ADD) *(modifiable + modIdx) += *(source + srcIdx);
	else
        if (op == OP_SUB) *(modifiable + modIdx) -= *(source + srcIdx);
	else
        if (op == OP_MUL) *(modifiable + modIdx) *= *(source + srcIdx);
	else 
	if (op == OP_SET) *(modifiable + modIdx)  = *(source + srcIdx);
    }
};

struct MatrixElementwiseVectorMultiplication
{
    complex_t* proxy;
    complex_t* Q_full;
    complex_t* V_full;
    pointInt_t size;
    __host__
    MatrixElementwiseVectorMultiplication (complex_t* proxy_,
		                           complex_t* Q_full_,
		                           complex_t* V_full_,
					   pointInt_t size_) : 
        proxy  (proxy_),
	Q_full (Q_full_),
	V_full (V_full_),
	size   (size_)
    {}

    __device__
    void operator () (int idx)
    {
	int idxV = idx % (size.x*size.y*4*size.z);
   
        *(proxy + idx) = *(Q_full + idx) * *(V_full + idxV);

    }    

};

struct FillS
{
    complex_t* destination;
    complex_t* acc;
    pointInt_t size;
    
    __host__
    FillS (complex_t * destination_,
           complex_t * acc_,
           pointInt_t size_) :
        destination (destination_),
        acc (acc_),
        size     (size_)
    {}

    __device__
    void operator()(int idx) const
    {
        int idxx = idx % size.x;
	idx /= size.x;
        int idxy = idx % size.y;
	int idxz = idx / size.y;
        *(destination + idxz*size.x*size.y + idxy*size.x + idxx) = 
        *(acc + idxz * 4*size.x*size.y + (size.x*2)*idxy + idxx) / (4.0f*size.x*size.y);
    }
};



struct MinusSourceUi
{
    point_t* deviceIndexesPtr;
    complex_t* source;
    complex_t* destination;
    const point_t sourcePos;
    const complex_t uiCoeff;
    __host__ 
    MinusSourceUi (point_t* deviceIndexesPtr_,
		   complex_t* source_,
		   complex_t* destination_,
		   const point_t sourcePos_,
		   const complex_t uiCoeff_) : 
        deviceIndexesPtr (deviceIndexesPtr_),
	source           (source_),
	destination      (destination_),
	sourcePos        (sourcePos_),
	uiCoeff          (uiCoeff_)
    {}

    __device__ 
    void operator () (int idx)
    { 
	point_t rec = *(deviceIndexesPtr + idx);
        point_t dr = {sourcePos.x - rec.x,
                      sourcePos.y - rec.y,
     		      sourcePos.z - rec.z};
        float len = dr.len ();
	complex_t value =  *(source + idx) * thrust::exp (uiCoeff * len) / (4 * 3.141592f * len);
	*(destination + idx) -= value;
    }

};

#undef CENTER_INDEX

#define CF(val) \
if ((cufft_error = val) != CUFFT_SUCCESS) \
printf ("ERROR on line %d, code %d\n", __LINE__, cufft_error);

struct MatVecFunctorFFT : MatVecFunctorBase
{
    complex_t* deviceDS2MatrixPtr;
    point_t * deviceIndexesPtr;
    int* seq;
    pointInt_t size;
    thrust::device_vector <complex_t> Q_full;
    thrust::device_vector <complex_t> V_full;
    thrust::device_vector <complex_t> result_proxy;
    thrust::device_vector <complex_t> accumulator;
    cufftHandle planQ;
    cufftHandle planV;
    int cufft_error;

    const float w2h3;
    const complex_t uiCoeff;
    const point_t sourcePos;
    __host__
    MatVecFunctorFFT (complex_t * deviceDS2MatrixPtr_,
                      point_t *   deviceIndexesPtr_,
                      int* seq_,
                      pointInt_t size_,
		      const float w2h3_,
		      const complex_t uiCoeff_,
		      const point_t sourcePos_) :
        deviceDS2MatrixPtr (deviceDS2MatrixPtr_),
        deviceIndexesPtr   (deviceIndexesPtr_),
        seq                (seq_),
        size               (size_),
	Q_full             (4*size.x*size.y*size.z*size.z, complex_t (0.0f, 0.0f)),
	V_full             (4*size.x*size.y*size.z,        complex_t (0.0f, 0.0f)),
	result_proxy       (4*size.x*size.y*size.z*size.z, complex_t (0.0f, 0.0f)),
	accumulator        (4*size.x*size.y*size.z,        complex_t (0.0f, 0.0f)),
	planQ              (),
	planV              (),
	cufft_error        (CUFFT_SUCCESS),
	w2h3               (w2h3_),
	uiCoeff            (uiCoeff_),
	sourcePos          (sourcePos_)
    {        
	int sizes[2] = {size.x*2, size.y*2};
        int gridSize = 4*size.x*size.y;
        CF(cufftPlanMany (&planQ, 2, sizes, nullptr, 1, gridSize, nullptr, 1, gridSize, CUFFT_C2C, size.z*size.z))
        CF(cufftPlanMany (&planV, 2, sizes, nullptr, 1, gridSize, nullptr, 1, gridSize, CUFFT_C2C, size.z))
        FillRadialQ fr (deviceIndexesPtr, Q_full.data ().get (), size, w2h3, uiCoeff);
        thrust::for_each (thrust::device, seq, seq + size.x*size.y*size.z*size.z, fr);
        CF(cufftExecC2C(planQ, 
                        reinterpret_cast<cufftComplex*> (Q_full.data ().get ()),
                        reinterpret_cast<cufftComplex*> (Q_full.data ().get ()), CUFFT_FORWARD))

    }

    __host__
    ~MatVecFunctorFFT ()
    {
        CF (cufftDestroy (planQ))
	CF (cufftDestroy (planV))
    }

    __host__
    void operator()(cuComplex* source, cuComplex* destination)
    {
        const int gridSize = 2*size.x*2*size.y;

#define TIME_PROFILE(x) 

        TIME_PROFILE(timespec ts[20] = {};)
	TIME_PROFILE(int ts_index = 0;)
        TIME_PROFILE(cudaDeviceSynchronize ();)
        TIME_PROFILE(clock_gettime(CLOCK_REALTIME, ts + ts_index); ts_index++;)//
	FillV fillV (deviceIndexesPtr, 
		     reinterpret_cast<complex_t*> (source),
		     V_full.data ().get (),
		     size,
		     sourcePos,
		     uiCoeff);
	
	thrust::for_each (thrust::device, seq, seq + size.x*size.y*size.z, fillV);
        
        TIME_PROFILE(cudaDeviceSynchronize ();)
        TIME_PROFILE(clock_gettime(CLOCK_REALTIME, ts + ts_index); ts_index++;)//
	CF(cufftExecC2C(planV, 
                        reinterpret_cast<cufftComplex*> (V_full.data ().get ()),
                        reinterpret_cast<cufftComplex*> (V_full.data ().get ()), CUFFT_FORWARD))
	

        TIME_PROFILE(cudaDeviceSynchronize ();)
        TIME_PROFILE(clock_gettime(CLOCK_REALTIME, ts + ts_index); ts_index++;)//
	MatrixElementwiseVectorMultiplication mevm (result_proxy.data ().get (),
			                            Q_full.data ().get (), 
						    V_full.data ().get (), 
						    size);

        thrust::for_each (thrust::device, seq, seq + 4*size.x*size.y*size.z*size.z, mevm);


        TIME_PROFILE(cudaDeviceSynchronize ();)
        TIME_PROFILE(clock_gettime(CLOCK_REALTIME, ts + ts_index); ts_index++;)//
        for (int q = 0; q < size.z; q++)
        {   
	    BlockElementwiseOperation lsf (accumulator.data ().get (), 
	                                   result_proxy.data ().get () + q*gridSize, (!q ? OP_SET : OP_ADD), size);
            thrust::for_each (thrust::device, seq, seq + gridSize*size.z, lsf);
        }
	
	TIME_PROFILE(cudaDeviceSynchronize ();)
	TIME_PROFILE(clock_gettime(CLOCK_REALTIME, ts + ts_index); ts_index++;)//
        CF(cufftExecC2C(planV, 
                        reinterpret_cast<cufftComplex*> (accumulator.data ().get ()),
                        reinterpret_cast<cufftComplex*> (accumulator.data ().get ()), CUFFT_INVERSE))
        
        TIME_PROFILE(cudaDeviceSynchronize ();)
        TIME_PROFILE(clock_gettime(CLOCK_REALTIME, ts + ts_index); ts_index++;)//
        FillS fs (reinterpret_cast<complex_t*> (destination), accumulator.data ().get (), size);
        thrust::for_each (thrust::device, seq, seq + size.x*size.y*size.z, fs);	

        TIME_PROFILE(cudaDeviceSynchronize ();)
        TIME_PROFILE(clock_gettime(CLOCK_REALTIME, ts + ts_index); ts_index++;)//
	
	ElementwiseOperation ds2_mul (reinterpret_cast<complex_t*> (destination), deviceDS2MatrixPtr, OP_MUL);
	thrust::for_each (thrust::device, seq, seq + size.x*size.y*size.z, ds2_mul);
        
        TIME_PROFILE(cudaDeviceSynchronize ();)
        TIME_PROFILE(clock_gettime(CLOCK_REALTIME, ts + ts_index); ts_index++;)//
        MinusSourceUi msu (deviceIndexesPtr, 
			   reinterpret_cast<complex_t*> (source),
			   reinterpret_cast<complex_t*> (destination),
			   sourcePos,
			   uiCoeff);
        thrust::for_each (thrust::device, seq, seq + size.x*size.y*size.z, msu);
        TIME_PROFILE(cudaDeviceSynchronize ();)
        TIME_PROFILE(clock_gettime(CLOCK_REALTIME, ts + ts_index); ts_index++;)//
	TIME_PROFILE(for (int i = 1; i < ts_index; i++)
	    printf ("%d->%d took %f mks\n", i - 1, i, 
	            ((ts[i].tv_sec - ts[i-1].tv_sec)*1000000000.0f + ts[i].tv_nsec - ts[i-1].tv_nsec)/1000.0f);)
    }
   #undef TIME_PROFILE
};

#undef CF

#else


struct SetAMatrix
{
    complex_t * deviceKMatrixPtr;
    point_t * deviceIndexesPtr;
    const point_t sourcePos;
    const complex_t uiCoeff;
    const int size3;
    SetAMatrix (complex_t * deviceKMatrixPtr_, 
		point_t * deviceIndexesPtr_,
		const point_t sourcePos_,
		const complex_t uiCoeff_,
		const int size3_) :
        deviceKMatrixPtr (deviceKMatrixPtr_),
        deviceIndexesPtr (deviceIndexesPtr_),
	sourcePos        (sourcePos_),
	uiCoeff          (uiCoeff_),
	size3            (size3_)
    {}

__device__
    complex_t operator() (int idx)
    {
        int idx1 = idx % size3; // receiver
        int idx2 = idx / size3;  // emitter
        if (idx1 == idx2) return complex_t (0.0f, 0.0f);
        
        point_t pos1 = *(deviceIndexesPtr + idx1);
        point_t pos2 = *(deviceIndexesPtr + idx2);
        point_t dr = {pos1.x-pos2.x,
                      pos1.y-pos2.y,
                      pos1.z-pos2.z};
        float len = dr.len ();

	
//--------------------------------------------------------------------+
// using ui in point   idx1   , maybe will need to tune               |
// if row-major order is used:                                        |
//                                  00 10 20                          |
//                                  01 11 21                          |
//                                  02 12 22                          |
//                                  03 13 23                          |
//                                  04 14 24                          |
//                                  05 15 25                          |
//                                  06 16 26                          |
//                                  07 17 27                          |
//                                  08 18 28                          |
//                                  09 19 29                          |
// every column contains all the points for a single receiver         |
// when converting to column-major:                                   |
// sequential receiver storage                                        |
//                                  00 01 02 03 04 05 06 07 08 09     |
//                                  10 11 12 13 14 15 16 17 18 19     |
//                                  20 21 22 23 24 25 26 27 28 29 ... |
//--------------------------------------------------------------------+


        return (*(deviceKMatrixPtr + idx2)) * thrust::exp (uiCoeff * len) / (4*3.141592f * len);
    }
};

struct ModifyAMatrix
{
    complex_t * deviceAMatrixPtr;
    point_t * deviceIndexesPtr;
    const point_t sourcePos;
    const int size3;
    const complex_t uiCoeff;

    ModifyAMatrix (complex_t * deviceAMatrixPtr_, 
                   point_t * deviceIndexesPtr_,
                   const point_t sourcePos_,
                   const int size3_,
                   const complex_t uiCoeff_) :
        deviceAMatrixPtr (deviceAMatrixPtr_),
        deviceIndexesPtr (deviceIndexesPtr_),
        sourcePos        (sourcePos_),
        size3            (size3_),
        uiCoeff          (uiCoeff_)
    {}

__device__
    void operator() (int idx)
    {
        point_t pos = *(deviceIndexesPtr + idx);
        point_t dr = {sourcePos.x - pos.x,
                      sourcePos.y - pos.y,
                      sourcePos.z - pos.z};
        float len = dr.len ();
        if (len < 0.0000001) return;
        *(deviceAMatrixPtr + idx*(size3 + 1)) = 
		-thrust::exp (uiCoeff * len) / (4 * 3.141592f * len);

    }
};
struct MatVecFunctor : MatVecFunctorBase
{
    cublasHandle_t cublasH;
    cuComplex* device_A_;
    size_t size_;
    
    __host__
    MatVecFunctor (cublasHandle_t cH,
                   complex_t * deviceAMatrixPtr,
                   size_t size) :
        cublasH   (cH),
        device_A_ (reinterpret_cast<cuComplex*> (deviceAMatrixPtr)),
        size_     (size)
    {}

    __host__
    void operator()(cuComplex* source, cuComplex* destination) 
    {
        cuComplex one = {1.0f, 0.0f};
        cuComplex zero = {0.0f, 0.0f};
        cublasCgemv (cublasH, CUBLAS_OP_N, size_, size_, &one,
                    device_A_, size_, source, 1, &zero, destination, 1);
    }
};

#endif
