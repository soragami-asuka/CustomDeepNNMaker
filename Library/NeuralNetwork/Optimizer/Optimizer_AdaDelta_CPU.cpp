//===============================================
// 最適化ルーチン(AdaDelta)
//===============================================
#include"stdafx.h"

#include<vector>

#include"Layer/NeuralNetwork/IOptimizer.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class Optimizer_AdaDelta_CPU : public iOptimizer_AdaDelta
	{
	private:
		U32 m_parameterCount;	/**< パラメータ数 */

		F32 m_rho;				/**< 減衰率. */
		F32 m_epsilon;			/**< 補助係数. */

		std::vector<F32> lpParameterH;
		std::vector<F32> lpParameterS;
		std::vector<F32> lpParameterV;

	public:
		/** コンストラクタ */
		Optimizer_AdaDelta_CPU(U32 i_parameterCount)
			:	m_parameterCount	(i_parameterCount)
			,	m_rho				(0.0f)
			,	m_epsilon			(0.0f)
		{
			this->lpParameterH.resize(m_parameterCount, 0.0f);
			this->lpParameterS.resize(m_parameterCount, 0.0f);
			this->lpParameterV.resize(m_parameterCount, 0.0f);
		}
		/** デストラクタ */
		virtual ~Optimizer_AdaDelta_CPU()
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
			for(U32 paramNum=0; paramNum<this->m_parameterCount; paramNum++)
			{
				// H更新
				this->lpParameterH[paramNum] = this->m_rho * this->lpParameterH[paramNum] + (1.0f - this->m_rho) * (i_lpDParameter[paramNum] * i_lpDParameter[paramNum]);

				// V更新
				this->lpParameterV[paramNum] = (sqrt(this->lpParameterS[paramNum] + this->m_epsilon)) *i_lpDParameter[paramNum] / (sqrt(this->lpParameterH[paramNum] + this->m_epsilon));

				// S更新
				this->lpParameterS[paramNum] = this->m_rho * this->lpParameterS[paramNum] + (1.0f - this->m_rho) * (this->lpParameterV[paramNum] * this->lpParameterV[paramNum]);

				// 重み更新
				io_lpParameter[paramNum] = io_lpParameter[paramNum] + this->lpParameterV[paramNum];
			}


			return ErrorCode::ERROR_CODE_NONE;
		}
	};

	/** オプティマイザを作成する */
	iOptimizer_AdaDelta* CreateOptimizer_AdaDelta_CPU(U32 i_parameterCount)
	{
		return new Optimizer_AdaDelta_CPU(i_parameterCount);
	}
	/** オプティマイザーを更新する.異なる型だった場合は強制的に指定の型に変換される. */
	ErrorCode UpdateOptimizer_AdaDelta_CPU(IOptimizer** io_ppOptimizer, U32 i_parameterCount, F32 i_rho, F32 i_epsilon)
	{
		iOptimizer_AdaDelta* pOptimizer = dynamic_cast<iOptimizer_AdaDelta*>(*io_ppOptimizer);
		if(pOptimizer == NULL)
		{
			if(*io_ppOptimizer)
				delete *io_ppOptimizer;

			*io_ppOptimizer = pOptimizer = CreateOptimizer_AdaDelta_CPU(i_parameterCount);
		}

		pOptimizer->SetHyperParameter(i_rho, i_epsilon);

		return ErrorCode::ERROR_CODE_NONE;
	}

}	// NeuralNetwork
}	// Layer
}	// Gravisbell