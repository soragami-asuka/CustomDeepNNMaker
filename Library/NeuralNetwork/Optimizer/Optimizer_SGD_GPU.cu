//===============================================
// 最適化ルーチン(SGD)
//===============================================
#include"stdafx.h"

#include"Optimizer_SGD_base.h"

#pragma warning(push)
#pragma warning(disable : 4267)
#include <cuda.h> // need CUDA_VERSION
#include <cudnn.h>
#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "device_launch_parameters.h"
#pragma warning(pop)


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class Optimizer_SGD_GPU : public Optimizer_SGD_base
	{
	private:
		cublasHandle_t cublasHandle;

	public:
		/** コンストラクタ */
		Optimizer_SGD_GPU(U32 i_parameterCount)
			:	Optimizer_SGD_base	(i_parameterCount)
		{
			cublasCreate(&cublasHandle);
		}
		/** デストラクタ */
		virtual ~Optimizer_SGD_GPU()
		{
			cublasDestroy(cublasHandle);
		}

	public:
		//===========================
		// 処理
		//===========================
		/** パラメータを更新する.
			@param io_lpParamter	更新するパラメータ.
			@param io_lpDParameter	パラメータの変化量. */
		ErrorCode UpdateParameter(F32 io_lpParameter[], const F32 i_lpDParameter[])
		{
			cublasSaxpy_v2(
				this->cublasHandle,
				this->m_parameterCount,
				&this->m_learnCoeff,
				i_lpDParameter,
				1,
				io_lpParameter,
				1);

			return ErrorCode::ERROR_CODE_NONE;
		}
	};

	/** オプティマイザを作成する */
	IOptimizer* CreateOptimizer_SGD_GPU(U32 i_parameterCount)
	{
		return new Optimizer_SGD_GPU(i_parameterCount);
	}
	/** オプティマイザをバッファから作成する */
	IOptimizer* CreateOptimizerFromBuffer_SGD_GPU(const BYTE* i_lpBuffer, Gravisbell::S32 i_bufferSize, Gravisbell::S32& o_useBufferSize)
	{
		return CreateOptimizerFromBuffer_SGD(i_lpBuffer, i_bufferSize, o_useBufferSize, CreateOptimizer_SGD_GPU);
	}
	/** オプティマイザーを更新する.異なる型だった場合は強制的に指定の型に変換される. */
	ErrorCode ChangeOptimizer_SGD_GPU(IOptimizer** io_ppOptimizer, U32 i_parameterCount)
	{
		Optimizer_SGD_GPU* pOptimizer = dynamic_cast<Optimizer_SGD_GPU*>(*io_ppOptimizer);
		if(pOptimizer == NULL)
		{
			if(*io_ppOptimizer)
				delete *io_ppOptimizer;

			*io_ppOptimizer = CreateOptimizer_SGD_GPU(i_parameterCount);
		}

		return ErrorCode::ERROR_CODE_NONE;
	}

}	// NeuralNetwork
}	// Layer
}	// Gravisbell