//===============================================
// 最適化ルーチン
//===============================================
#ifndef __GRAVISBELL_LIBRARY_NN_OPTIMIZER_H__
#define __GRAVISBELL_LIBRARY_NN_OPTIMIZER_H__

#include"Layer/NeuralNetwork/IOptimizer.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	/** オプティマイザーを作成する.[SGD] */
	iOptimizer_SGD* CreateOptimizer_SGD_CPU(U32 i_parameterCount);
	iOptimizer_SGD* CreateOptimizer_SGD_GPU(U32 i_parameterCount);

	/** オプティマイザーを更新する.異なる型だった場合は強制的に指定の型に変換される. */
	ErrorCode UpdateOptimizer_SGD_CPU(IOptimizer** io_ppOptimizer, U32 i_parameterCount, F32 i_learnCoeff);
	ErrorCode UpdateOptimizer_SGD_GPU(IOptimizer** io_ppOptimizer, U32 i_parameterCount, F32 i_learnCoeff);


	/** オプティマイザーを作成する.[Momentum] */
	iOptimizer_Momentum* CreateOptimizer_Momentum_CPU(U32 i_parameterCount);
	iOptimizer_Momentum* CreateOptimizer_Momentum_GPU(U32 i_parameterCount);

	/** オプティマイザーを更新する.異なる型だった場合は強制的に指定の型に変換される. */
	ErrorCode UpdateOptimizer_Momentum_CPU(IOptimizer** io_ppOptimizer, U32 i_parameterCount, F32 i_learnCoeff, F32 i_alpha);
	ErrorCode UpdateOptimizer_Momentum_GPU(IOptimizer** io_ppOptimizer, U32 i_parameterCount, F32 i_learnCoeff, F32 i_alpha);



	/** オプティマイザーを作成する.[AdaDelta] */
	iOptimizer_AdaDelta* CreateOptimizer_AdaDelta_CPU(U32 i_parameterCount);
	iOptimizer_AdaDelta* CreateOptimizer_AdaDelta_GPU(U32 i_parameterCount);

	/** オプティマイザーを作成する.[Adam] */
	iOptimizer_Adam* CreateOptimizer_Adam_CPU(U32 i_parameterCount);
	iOptimizer_Adam* CreateOptimizer_Adam_GPU(U32 i_parameterCount);

}	// NeuralNetwork
}	// Layer
}	// Gravisbell


#endif
