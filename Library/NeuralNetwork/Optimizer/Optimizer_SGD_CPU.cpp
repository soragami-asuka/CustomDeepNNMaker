//===============================================
// 最適化ルーチン(SGD)
//===============================================
#include"stdafx.h"

#include<stdio.h>

#include"Layer/NeuralNetwork/IOptimizer.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class Optimizer_SGD_CPU : public iOptimizer_SGD
	{
	private:
		U32 m_parameterCount;	/**< パラメータ数 */
		F32 m_learnCoeff;	/**< 学習係数 */

	public:
		/** コンストラクタ */
		Optimizer_SGD_CPU(U32 i_parameterCount)
			:	m_parameterCount	(i_parameterCount)
			,	m_learnCoeff		(0.0f)
		{
		}
		/** デストラクタ */
		virtual ~Optimizer_SGD_CPU()
		{
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
			for(U32 paramNum=0; paramNum<this->m_parameterCount; paramNum++)
			{
				io_lpParameter[paramNum] += this->m_learnCoeff * i_lpDParameter[paramNum];
			}

			return ErrorCode::ERROR_CODE_NONE;
		}
	};

	/** オプティマイザを作成する */
	iOptimizer_SGD* CreateOptimizer_SGD_CPU(U32 i_parameterCount)
	{
		return new Optimizer_SGD_CPU(i_parameterCount);
	}
	/** オプティマイザーを更新する.異なる型だった場合は強制的に指定の型に変換される. */
	ErrorCode UpdateOptimizer_SGD_CPU(IOptimizer** io_ppOptimizer, U32 i_parameterCount, F32 i_learnCoeff)
	{
		iOptimizer_SGD* pOptimizer = dynamic_cast<iOptimizer_SGD*>(*io_ppOptimizer);
		if(pOptimizer == NULL)
		{
			if(*io_ppOptimizer)
				delete *io_ppOptimizer;

			*io_ppOptimizer = pOptimizer = CreateOptimizer_SGD_CPU(i_parameterCount);
		}

		pOptimizer->SetHyperParameter(i_learnCoeff);

		return ErrorCode::ERROR_CODE_NONE;
	}

}	// NeuralNetwork
}	// Layer
}	// Gravisbell