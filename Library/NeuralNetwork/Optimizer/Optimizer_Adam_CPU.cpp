//===============================================
// 最適化ルーチン(Adam)
//===============================================
#include"stdafx.h"

#include<vector>

#include"Layer/NeuralNetwork/IOptimizer.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class Optimizer_Adam_CPU : public iOptimizer_Adam
	{
	private:
		U32 m_parameterCount;	/**< パラメータ数 */

		F32	m_alpha;		/**< 慣性. */
		F32	m_beta1;		/**< 減衰率. */
		F32	m_beta2;		/**< 減衰率. */
		F32	m_epsilon;		/**< 補助係数. */

		std::vector<F32> lpParameterM;
		std::vector<F32> lpParameterV;

		F32 m_beta1Pows;	/**< β1の階乗値 */
		F32 m_beta2Pows;	/**< β2の階乗値 */

	public:
		/** コンストラクタ */
		Optimizer_Adam_CPU(U32 i_parameterCount)
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
		virtual ~Optimizer_Adam_CPU()
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

			if(m_beta1Pows < 0.0f)
				m_beta1Pows = m_beta1;
			if(m_beta2Pows < 0.0f)
				m_beta2Pows = m_beta2;


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

			for(U32 paramNum=0; paramNum<this->m_parameterCount; paramNum++)
			{
				this->lpParameterM[paramNum] = this->m_beta1 * this->lpParameterM[paramNum] + (1.0f - this->m_beta1) * i_lpDParameter[paramNum];
				this->lpParameterV[paramNum] = this->m_beta2 * this->lpParameterV[paramNum] + (1.0f - this->m_beta2) * i_lpDParameter[paramNum] * i_lpDParameter[paramNum];

				F32 tmpM = this->lpParameterM[paramNum] / (1.0f - this->m_beta1Pows);
				F32 tmpV = this->lpParameterV[paramNum] / (1.0f - this->m_beta2Pows);

				io_lpParameter[paramNum] += this->m_alpha * (tmpM / (sqrt(tmpV) + this->m_epsilon));
			}

			return ErrorCode::ERROR_CODE_NONE;
		}
	};

	/** オプティマイザを作成する */
	iOptimizer_Adam* CreateOptimizer_Adam_CPU(U32 i_parameterCount)
	{
		return new Optimizer_Adam_CPU(i_parameterCount);
	}
	/** オプティマイザーを更新する.異なる型だった場合は強制的に指定の型に変換される. */
	ErrorCode UpdateOptimizer_Adam_CPU(IOptimizer** io_ppOptimizer, U32 i_parameterCount, F32 i_alpha, F32 i_beta1, F32 i_beta2, F32 i_epsilon)
	{
		iOptimizer_Adam* pOptimizer = dynamic_cast<iOptimizer_Adam*>(*io_ppOptimizer);
		if(pOptimizer == NULL)
		{
			if(*io_ppOptimizer)
				delete *io_ppOptimizer;

			*io_ppOptimizer = pOptimizer = CreateOptimizer_Adam_CPU(i_parameterCount);
		}

		pOptimizer->SetHyperParameter(i_alpha, i_beta1, i_beta2, i_epsilon);

		return ErrorCode::ERROR_CODE_NONE;
	}

}	// NeuralNetwork
}	// Layer
}	// Gravisbell