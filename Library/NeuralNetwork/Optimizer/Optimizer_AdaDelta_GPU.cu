//===============================================
// 最適化ルーチン(AdaDelta)
//===============================================
#include"stdafx.h"

#include<vector>

#include"Layer/NeuralNetwork/IOptimizer.h"

// CUDA用
#pragma warning(push)
#pragma warning(disable : 4267)
#include <cuda.h>
#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "device_launch_parameters.h"
#pragma warning(pop)


#define BLOCK_SIZE	(32)


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	namespace
	{
		/** ベクトルの要素同士の掛け算. */
		__global__ void cuda_func_updateParameter(F32* io_lpParameter, const F32* i_lpDParameter, const U32 i_bufferSize, F32* io_lpParameterH, F32* io_lpParameterS, F32* io_lpParameterV, F32 i_rho, F32 i_epsilon)
		{
			const U32 paramNum = blockIdx.x * BLOCK_SIZE + threadIdx.x;
			if(paramNum >= i_bufferSize)	// 分岐するが末尾のwarpだけなので、処理速度に影響はないはず...
				return;

			// H更新
			io_lpParameterH[paramNum] = i_rho * io_lpParameterH[paramNum] + (1.0f - i_rho) * (i_lpDParameter[paramNum] * i_lpDParameter[paramNum]);

			// V更新
			io_lpParameterV[paramNum] = (sqrtf(io_lpParameterS[paramNum] + i_epsilon)) *i_lpDParameter[paramNum] / (sqrtf(io_lpParameterH[paramNum] + i_epsilon));

			// S更新
			io_lpParameterS[paramNum] = i_rho * io_lpParameterS[paramNum] + (1.0f - i_rho) * (io_lpParameterV[paramNum] * io_lpParameterV[paramNum]);

			// 重み更新
			io_lpParameter[paramNum] = io_lpParameter[paramNum] + io_lpParameterV[paramNum];
		}
	}


	class Optimizer_AdaDelta_GPU : public iOptimizer_AdaDelta
	{
	private:
		U32 m_parameterCount;	/**< パラメータ数 */

		F32 m_rho;				/**< 減衰率. */
		F32 m_epsilon;			/**< 補助係数. */

		thrust::device_vector<F32> lpParameterH;
		thrust::device_vector<F32> lpParameterS;
		thrust::device_vector<F32> lpParameterV;

	public:
		/** コンストラクタ */
		Optimizer_AdaDelta_GPU(U32 i_parameterCount)
			:	m_parameterCount	(i_parameterCount)
			,	m_rho				(0.0f)
			,	m_epsilon			(0.0f)
		{
			this->lpParameterH.resize(m_parameterCount, 0.0f);
			this->lpParameterS.resize(m_parameterCount, 0.0f);
			this->lpParameterV.resize(m_parameterCount, 0.0f);
		}
		/** デストラクタ */
		virtual ~Optimizer_AdaDelta_GPU()
		{
		}

	public:
		//===========================
		// 基本情報
		//===========================
		/** オプティマイザの種別を取得する */
		OptimizerType GetTypeCode()const
		{
			return OptimizerType::OPTIMIZER_TYPE_ADADELTA;
		}
		
		/** ハイパーパラメータを更新する
			@param	i_rho			減衰率.
			@param	i_epsilon		補助係数. */
		virtual ErrorCode SetHyperParameter(F32 i_rho, F32 i_epsilon)
		{
			this->m_rho    = i_rho;
			this->m_epsilon = i_epsilon;

			return ErrorCode::ERROR_CODE_NONE;
		}


		//===========================
		// 処理
		//===========================
		/** パラメータを更新する.
			@param io_lpParamter	更新するパラメータ.
			@param io_lpDParameter	パラメータの変化量. */
		ErrorCode UpdateParameter(F32 io_lpParameter[], const F32 i_lpDParameter[])
		{
			dim3 grid((this->m_parameterCount +(BLOCK_SIZE - 1))/BLOCK_SIZE , 1, 1);
			dim3 block(BLOCK_SIZE, 1, 1);

			cuda_func_updateParameter<<<grid, block>>>(
				io_lpParameter,
				i_lpDParameter,
				this->m_parameterCount,
				thrust::raw_pointer_cast(&this->lpParameterH[0]),
				thrust::raw_pointer_cast(&this->lpParameterS[0]),
				thrust::raw_pointer_cast(&this->lpParameterV[0]),
				this->m_rho,
				this->m_epsilon);


			return ErrorCode::ERROR_CODE_NONE;
		}
	};

	/** オプティマイザを作成する */
	iOptimizer_AdaDelta* CreateOptimizer_AdaDelta_GPU(U32 i_parameterCount)
	{
		return new Optimizer_AdaDelta_GPU(i_parameterCount);
	}
	/** オプティマイザーを更新する.異なる型だった場合は強制的に指定の型に変換される. */
	ErrorCode UpdateOptimizer_AdaDelta_GPU(IOptimizer** io_ppOptimizer, U32 i_parameterCount, F32 i_rho, F32 i_epsilon)
	{
		iOptimizer_AdaDelta* pOptimizer = dynamic_cast<iOptimizer_AdaDelta*>(*io_ppOptimizer);
		if(pOptimizer == NULL)
		{
			if(*io_ppOptimizer)
				delete *io_ppOptimizer;

			*io_ppOptimizer = pOptimizer = CreateOptimizer_AdaDelta_GPU(i_parameterCount);
		}

		pOptimizer->SetHyperParameter(i_rho, i_epsilon);

		return ErrorCode::ERROR_CODE_NONE;
	}

}	// NeuralNetwork
}	// Layer
}	// Gravisbell