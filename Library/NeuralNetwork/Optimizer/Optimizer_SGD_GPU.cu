//===============================================
// 最適化ルーチン(SGD)
//===============================================
#include"stdafx.h"

#include"Layer/NeuralNetwork/IOptimizer.h"

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

	class Optimizer_SGD_GPU : public iOptimizer_SGD
	{
	private:
		U32 m_parameterCount;	/**< パラメータ数 */
		F32 m_learnCoeff;	/**< 学習係数 */

		cublasHandle_t cublasHandle;

	public:
		/** コンストラクタ */
		Optimizer_SGD_GPU(U32 i_parameterCount)
			:	m_parameterCount	(i_parameterCount)
			,	m_learnCoeff		(0.0f)
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
		// 基本情報
		//===========================
		/** オプティマイザの種別を取得する */
		OptimizerType GetTypeCode()const
		{
			return OptimizerType::OPTIMIZER_TYPE_SGD;
		}
		
		/** ハイパーパラメータを更新する
			@param	i_learnCoeff	学習係数 */
		ErrorCode SetHyperParameter(F32 i_learnCoeff)
		{
			this->m_learnCoeff = i_learnCoeff;

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
	iOptimizer_SGD* CreateOptimizer_SGD_GPU(U32 i_parameterCount)
	{
		return new Optimizer_SGD_GPU(i_parameterCount);
	}
	/** オプティマイザーを更新する.異なる型だった場合は強制的に指定の型に変換される. */
	ErrorCode UpdateOptimizer_SGD_GPU(IOptimizer** io_ppOptimizer, U32 i_parameterCount, F32 i_learnCoeff)
	{
		iOptimizer_SGD* pOptimizer = dynamic_cast<iOptimizer_SGD*>(*io_ppOptimizer);
		if(pOptimizer == NULL)
		{
			if(*io_ppOptimizer)
				delete *io_ppOptimizer;

			*io_ppOptimizer = pOptimizer = CreateOptimizer_SGD_GPU(i_parameterCount);
		}

		pOptimizer->SetHyperParameter(i_learnCoeff);

		return ErrorCode::ERROR_CODE_NONE;
	}

}	// NeuralNetwork
}	// Layer
}	// Gravisbell