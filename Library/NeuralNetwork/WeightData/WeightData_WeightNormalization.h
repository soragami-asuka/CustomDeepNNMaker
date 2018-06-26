//=====================================
// 重みデータクラス.
// デフォルト.
//=====================================
#ifndef __GRAVISBELL_NN_WEIGHTDATA_WEIGHTNORMALIZATION_H__
#define __GRAVISBELL_NN_WEIGHTDATA_WEIGHTNORMALIZATION_H__

#include"Layer/NeuralNetwork/IWeightData.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	/** 重みクラスを作成する.
		デフォルト.CPU制御. */
	IWeightData* CreateWeightData_WeightNormalization_CPU(U32 i_neuronCount, U32 i_inputCount);
	/** 重みクラスを作成する.
		デフォルト.GPU制御. */
	IWeightData* CreateWeightData_WeightNormalization_GPU(U32 i_neuronCount, U32 i_inputCount);

}	// NeuralNetwork
}	// Layer
}	// Gravisbell

#endif	__GRAVISBELL_NN_INITIALIZER_ZERO_H__