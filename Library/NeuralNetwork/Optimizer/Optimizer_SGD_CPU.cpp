//===============================================
// 最適化ルーチン(SGD)
//===============================================
#include"stdafx.h"

#include<stdio.h>

#include"Optimizer_SGD_base.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class Optimizer_SGD_CPU : public Optimizer_SGD_base
	{
	public:
		/** コンストラクタ */
		Optimizer_SGD_CPU(U32 i_parameterCount)
			:	Optimizer_SGD_base	(i_parameterCount)
		{
		}
		/** デストラクタ */
		virtual ~Optimizer_SGD_CPU()
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
			for(U32 paramNum=0; paramNum<this->m_parameterCount; paramNum++)
			{
				io_lpParameter[paramNum] += this->m_learnCoeff * i_lpDParameter[paramNum];
			}

			return ErrorCode::ERROR_CODE_NONE;
		}
	};

	/** オプティマイザを作成する */
	IOptimizer* CreateOptimizer_SGD_CPU(U32 i_parameterCount)
	{
		return new Optimizer_SGD_CPU(i_parameterCount);
	}
	/** オプティマイザをバッファから作成する */
	IOptimizer* CreateOptimizerFromBuffer_SGD_CPU(const BYTE* i_lpBuffer, Gravisbell::S32 i_bufferSize, Gravisbell::S32& o_useBufferSize)
	{
		return CreateOptimizerFromBuffer_SGD(i_lpBuffer, i_bufferSize, o_useBufferSize, CreateOptimizer_SGD_CPU);
	}
	/** オプティマイザーを更新する.異なる型だった場合は強制的に指定の型に変換される. */
	ErrorCode ChangeOptimizer_SGD_CPU(IOptimizer** io_ppOptimizer, U32 i_parameterCount)
	{
		Optimizer_SGD_CPU* pOptimizer = dynamic_cast<Optimizer_SGD_CPU*>(*io_ppOptimizer);
		if(pOptimizer == NULL)
		{
			if(*io_ppOptimizer)
				delete *io_ppOptimizer;

			*io_ppOptimizer = CreateOptimizer_SGD_CPU(i_parameterCount);
		}

		return ErrorCode::ERROR_CODE_NONE;
	}

}	// NeuralNetwork
}	// Layer
}	// Gravisbell