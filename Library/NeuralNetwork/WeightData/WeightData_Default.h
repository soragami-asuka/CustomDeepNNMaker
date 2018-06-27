//=====================================
// �d�݃f�[�^�N���X.
// �f�t�H���g.
//=====================================
#ifndef __GRAVISBELL_NN_WEIGHTDATA_DEFAULT_H__
#define __GRAVISBELL_NN_WEIGHTDATA_DEFAULT_H__

#include"Layer/NeuralNetwork/IWeightData.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	/** �d�݃N���X���쐬����.
		�f�t�H���g.CPU����. */
	IWeightData* CreateWeightData_Default_CPU(U32 i_neuronCount, U32 i_inputCount);
	/** �d�݃N���X���쐬����.
		�f�t�H���g.GPU����. */
	IWeightData* CreateWeightData_Default_GPU(U32 i_neuronCount, U32 i_inputCount);

}	// NeuralNetwork
}	// Layer
}	// Gravisbell

#endif	__GRAVISBELL_NN_INITIALIZER_ZERO_H__