//===============================================
// 最適化ルーチン
//===============================================
#ifndef __GRAVISBELL_LIBRARY_NN_OPTIMIZER_H__
#define __GRAVISBELL_LIBRARY_NN_OPTIMIZER_H__

#ifdef OPTIMIZER_EXPORTS
#define Optimizer_API __declspec(dllexport)
#else
#define Optimizer_API __declspec(dllimport)
#ifndef GRAVISBELL_LIBRARY
#endif
#endif

#include"Layer/NeuralNetwork/IOptimizer.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	/** オプティマイザーを変更する */
	Optimizer_API ErrorCode ChangeOptimizer_CPU(IOptimizer** io_ppOptimizer, const wchar_t i_optimizerID[], U32 i_parameterCount);
	Optimizer_API ErrorCode ChangeOptimizer_GPU(IOptimizer** io_ppOptimizer, const wchar_t i_optimizerID[], U32 i_parameterCount);

	/** オプティマイザーをバッファから作成する */
	Optimizer_API IOptimizer* CreateOptimizerFromBuffer_CPU(const BYTE* i_lpBuffer, Gravisbell::S32 i_bufferSize, Gravisbell::S32& o_useBufferSize);
	Optimizer_API IOptimizer* CreateOptimizerFromBuffer_GPU(const BYTE* i_lpBuffer, Gravisbell::S32 i_bufferSize, Gravisbell::S32& o_useBufferSize);

}	// NeuralNetwork
}	// Layer
}	// Gravisbell


#endif
