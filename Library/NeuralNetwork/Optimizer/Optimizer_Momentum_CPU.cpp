//===============================================
// 最適化ルーチン(Momentum)
//===============================================
#include"stdafx.h"

#include<vector>

#include"Layer/NeuralNetwork/IOptimizer.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class Optimizer_Momentum_CPU : public iOptimizer_Momentum
	{
	private:
		U32 m_parameterCount;	/**< パラメータ数 */
		F32 m_learnCoeff;		/**< 学習係数 */
		F32 m_alpha;				/**< 慣性項 */

		std::vector<F32> m_lpLastDParameter;	/**< 直前の更新の際のパラメータ変化量 */

	public:
		/** コンストラクタ */
		Optimizer_Momentum_CPU(U32 i_parameterCount)
			:	m_parameterCount	(i_parameterCount)
			,	m_learnCoeff		(0.0f)
			,	m_alpha				(0.0f)
		{
			this->m_lpLastDParameter.resize(m_parameterCount, 0.0f);
		}
		/** デストラクタ */
		virtual ~Optimizer_Momentum_CPU()
		{
		}

	public:
		//===========================
		// 基本情報
		//===========================
		/** オプティマイザの種別を取得する */
		OptimizerType GetTypeCode()const
		{
			return OptimizerType::OPTIMIZER_TYPE_MOMENTUM;
		}
		
		/** ハイパーパラメータを更新する
			@param	i_learnCoeff	学習係数
			@param	i_alpha			慣性. */
		virtual ErrorCode SetHyperParameter(F32 i_learnCoeff, F32 i_alpha)
		{
			this->m_learnCoeff = i_learnCoeff;
			this->m_alpha = i_alpha;

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
			// 変化量を更新
			for(U32 paramNum=0; paramNum<this->m_parameterCount; paramNum++)
			{
				this->m_lpLastDParameter[paramNum] = this->m_alpha * this->m_lpLastDParameter[paramNum] + this->m_learnCoeff * i_lpDParameter[paramNum];
			}

			// パラメータ更新
			for(U32 paramNum=0; paramNum<this->m_parameterCount; paramNum++)
			{
				io_lpParameter[paramNum] += m_lpLastDParameter[paramNum];
			}

			return ErrorCode::ERROR_CODE_NONE;
		}
	};

	/** オプティマイザを作成する */
	iOptimizer_Momentum* CreateOptimizer_Momentum_CPU(U32 i_parameterCount)
	{
		return new Optimizer_Momentum_CPU(i_parameterCount);
	}
	/** オプティマイザーを更新する.異なる型だった場合は強制的に指定の型に変換される. */
	ErrorCode UpdateOptimizer_Momentum_CPU(IOptimizer** io_ppOptimizer, U32 i_parameterCount, F32 i_learnCoeff, F32 i_alpha)
	{
		iOptimizer_Momentum* pOptimizer = dynamic_cast<iOptimizer_Momentum*>(*io_ppOptimizer);
		if(pOptimizer == NULL)
		{
			if(*io_ppOptimizer)
				delete *io_ppOptimizer;

			*io_ppOptimizer = pOptimizer = CreateOptimizer_Momentum_CPU(i_parameterCount);
		}

		pOptimizer->SetHyperParameter(i_learnCoeff, i_alpha);

		return ErrorCode::ERROR_CODE_NONE;
	}

}	// NeuralNetwork
}	// Layer
}	// Gravisbell