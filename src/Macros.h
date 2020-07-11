#include "CudaError.h"

#define CUDA_CALL(err) ErrorHelper::cudaSafeCall(err,__FILE__,__LINE__);
#define CUDA_CHECK_ERROR CUDA_CALL(cudaGetLastError())