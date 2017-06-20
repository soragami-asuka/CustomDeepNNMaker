//===============================================
// 最適化ルーチン(Adam)
//===============================================
#include"stdafx.h"

#include<vector>

#include"Optimizer_Adam_base.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class Optimizer_Adam_CPU : public Optimizer_Adam_base
	{
	public:
		std::vector<F32> lpParameterM;
		std::vector<F32> lpParameterV;

		F32 m_beta1Pows;	/**< β1の階乗値 */
		F32 m_beta2Pows;	/**< β2の階乗値 */

	public:
		/** コンストラクタ */
		Optimizer_Adam_CPU(U32 i_parameterCount)
			:	Optimizer_Adam_base	(i_parameterCount)
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

	public:
		//===========================
		// 保存
		//===========================
		/** レイヤーをバッファに書き込む.
			@param o_lpBuffer	書き込み先バッファの先頭アドレス. GetUseBufferByteCountの戻り値のバイト数が必要
			@return 成功した場合書き込んだバッファサイズ.失敗した場合は負の値 */
		S32 WriteToBuffer(BYTE* o_lpBuffer)const
		{
			U32 writePos = WriteToBufferBase(o_lpBuffer);

			// M
			memcpy(&o_lpBuffer[writePos], &this->lpParameterM[0], sizeof(F32)*this->m_parameterCount);
			writePos += sizeof(F32)*this->m_parameterCount;
			// V
			memcpy(&o_lpBuffer[writePos], &this->lpParameterV[0], sizeof(F32)*this->m_parameterCount);
			writePos += sizeof(F32)*this->m_parameterCount;

			// beta1^n
			memcpy(&o_lpBuffer[writePos], &this->m_beta1Pows, sizeof(F32));
			writePos += sizeof(F32);
			// beta2^n
			memcpy(&o_lpBuffer[writePos], &this->m_beta2Pows, sizeof(F32));
			writePos += sizeof(F32);

			return writePos;
		}
	};

	/** オプティマイザを作成する */
	Optimizer_Adam_base* CreateOptimizer_Adam_CPU(U32 i_parameterCount)
	{
		return new Optimizer_Adam_CPU(i_parameterCount);
	}
	/** オプティマイザをバッファから作成する */
	IOptimizer* CreateOptimizerFromBuffer_Adam_CPU(const BYTE* i_lpBuffer, Gravisbell::S32 i_bufferSize, Gravisbell::S32& o_useBufferSize)
	{
		Optimizer_Adam_base* pOptimizer = CreateOptimizerFromBuffer_Adam(i_lpBuffer, i_bufferSize, o_useBufferSize, CreateOptimizer_Adam_CPU);
		if(pOptimizer == NULL)
			return NULL;
		Optimizer_Adam_CPU* pOptimizerCPU = dynamic_cast<Optimizer_Adam_CPU*>(pOptimizer);
		if(pOptimizerCPU == NULL)
		{
			delete pOptimizer;
			return NULL;
		}

		// M
		memcpy(&pOptimizerCPU->lpParameterM[0], &i_lpBuffer[o_useBufferSize], sizeof(F32)*pOptimizerCPU->lpParameterM.size());
		o_useBufferSize += sizeof(F32)*pOptimizerCPU->lpParameterM.size();
		// V
		memcpy(&pOptimizerCPU->lpParameterV[0], &i_lpBuffer[o_useBufferSize], sizeof(F32)*pOptimizerCPU->lpParameterV.size());
		o_useBufferSize += sizeof(F32)*pOptimizerCPU->lpParameterV.size();

		// beta1^n
		memcpy(&pOptimizerCPU->m_beta1Pows, &i_lpBuffer[o_useBufferSize], sizeof(F32));
		o_useBufferSize += sizeof(F32);
		// beta2^n
		memcpy(&pOptimizerCPU->m_beta2Pows, &i_lpBuffer[o_useBufferSize], sizeof(F32));
		o_useBufferSize += sizeof(F32);

		return pOptimizer;
	}
	/** オプティマイザーを更新する.異なる型だった場合は強制的に指定の型に変換される. */
	ErrorCode ChangeOptimizer_Adam_CPU(IOptimizer** io_ppOptimizer, U32 i_parameterCount)
	{
		Optimizer_Adam_base* pOptimizer = dynamic_cast<Optimizer_Adam_base*>(*io_ppOptimizer);
		if(pOptimizer == NULL)
		{
			if(*io_ppOptimizer)
				delete *io_ppOptimizer;

			*io_ppOptimizer = CreateOptimizer_Adam_CPU(i_parameterCount);
		}

		return ErrorCode::ERROR_CODE_NONE;
	}

}	// NeuralNetwork
}	// Layer
}	// Gravisbell