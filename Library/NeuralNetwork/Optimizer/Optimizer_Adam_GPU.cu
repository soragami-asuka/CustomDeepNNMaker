//===============================================
// 最適化ルーチン(Adam)
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
		__global__ void cuda_func_updateParameter(F32* io_lpParameter, const F32* i_lpDParameter, const U32 i_bufferSize, F32* io_lpParameterM, F32* io_lpParameterV, F32 i_alpha, F32 i_beta1, F32 i_beta2, F32 i_epsilon, F32 i_beta1Pows, F32 i_beta2Pows)
		{
			const U32 paramNum = blockIdx.x * BLOCK_SIZE + threadIdx.x;
			if(paramNum >= i_bufferSize)	// 分岐するが末尾のwarpだけなので、処理速度に影響はないはず...
				return;

			io_lpParameterM[paramNum] = i_beta1 * io_lpParameterM[paramNum] + (1.0f - i_beta1) * i_lpDParameter[paramNum];
			io_lpParameterV[paramNum] = i_beta2 * io_lpParameterV[paramNum] + (1.0f - i_beta2) * i_lpDParameter[paramNum] * i_lpDParameter[paramNum];

			F32 tmpM = io_lpParameterM[paramNum] / (1.0f - i_beta1Pows);
			F32 tmpV = io_lpParameterV[paramNum] / (1.0f - i_beta2Pows);

			io_lpParameter[paramNum] += i_alpha * (tmpM / (sqrt(tmpV) + i_epsilon));
		}
	}

	class Optimizer_Adam_GPU : public iOptimizer_Adam
	{
	private:
		U32 m_parameterCount;	/**< パラメータ数 */

		F32	m_alpha;		/**< 慣性. */
		F32	m_beta1;		/**< 減衰率. */
		F32	m_beta2;		/**< 減衰率. */
		F32	m_epsilon;		/**< 補助係数. */

		thrust::device_vector<F32> lpParameterM;
		thrust::device_vector<F32> lpParameterV;

		F32 m_beta1Pows;	/**< β1の階乗値 */
		F32 m_beta2Pows;	/**< β2の階乗値 */

	public:
		/** コンストラクタ */
		Optimizer_Adam_GPU(U32 i_parameterCount)
			:	m_parameterCount	(i_parameterCount)
			,	m_alpha		(0.0f)
			,	m_beta1		(0.0f)
			,	m_beta2		(0.0f)
			,	m_epsilon	(0.0f)
			,	m_beta2Pows	(1.0f)	/**< β2の階乗値 */
			,	m_beta1Pows	(1.0f)	/**< β1の階乗値 */
		{
			this->lpParameterM.resize(this->m_parameterCount);
			this->lpParameterV.resize(this->m_parameterCount);
		}
		/** デストラクタ */
		virtual ~Optimizer_Adam_GPU()
		{
		}

	public:
		//===========================
		// 基本情報
		//===========================
		/** オプティマイザの種別を取得する */
		OptimizerType GetTypeCode()const
		{
			return OptimizerType::OPTIMIZER_TYPE_ADAM;
		}
		
		/** ハイパーパラメータを更新する
			@param	i_alpha			慣性.
			@param	i_beta1			減衰率.
			@param	i_beta2			減衰率.
			@param	i_epsilon		補助係数. */
		virtual ErrorCode SetHyperParameter(F32 i_alpha, F32 i_beta1, F32 i_beta2, F32 i_epsilon)
		{
			m_alpha   = i_alpha;
			m_beta1   = i_beta1;
			m_beta2	  = i_beta2;
			m_epsilon = i_epsilon;

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
			this->m_beta1Pows *= this->m_beta1;
			this->m_beta2Pows *= this->m_beta2;
			
			dim3 grid((this->m_parameterCount +(BLOCK_SIZE - 1))/BLOCK_SIZE , 1, 1);
			dim3 block(BLOCK_SIZE, 1, 1);

			cuda_func_updateParameter<<<grid, block>>>(
				io_lpParameter,
				i_lpDParameter,
				this->m_parameterCount,
				thrust::raw_pointer_cast(&this->lpParameterM[0]),
				thrust::raw_pointer_cast(&this->lpParameterV[0]),
				this->m_alpha, this->m_beta1, this->m_beta2, this->m_epsilon,
				this->m_beta1Pows, this->m_beta2Pows);


			return ErrorCode::ERROR_CODE_NONE;
		}
	};

	/** オプティマイザを作成する */
	iOptimizer_Adam* CreateOptimizer_Adam_GPU(U32 i_parameterCount)
	{
		return new Optimizer_Adam_GPU(i_parameterCount);
	}
	/** オプティマイザーを更新する.異なる型だった場合は強制的に指定の型に変換される. */
	ErrorCode UpdateOptimizer_Adam_GPU(IOptimizer** io_ppOptimizer, U32 i_parameterCount, F32 i_alpha, F32 i_beta1, F32 i_beta2, F32 i_epsilon)
	{
		iOptimizer_Adam* pOptimizer = dynamic_cast<iOptimizer_Adam*>(*io_ppOptimizer);
		if(pOptimizer == NULL)
		{
			if(*io_ppOptimizer)
				delete *io_ppOptimizer;

			*io_ppOptimizer = pOptimizer = CreateOptimizer_Adam_GPU(i_parameterCount);
		}

		pOptimizer->SetHyperParameter(i_alpha, i_beta1, i_beta2, i_epsilon);

		return ErrorCode::ERROR_CODE_NONE;
	}

}	// NeuralNetwork
}	// Layer
}	// Gravisbell