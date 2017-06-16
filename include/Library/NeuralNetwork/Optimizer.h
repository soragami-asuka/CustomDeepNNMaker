//===============================================
// �œK�����[�`��
//===============================================
#ifndef __GRAVISBELL_LIBRARY_NN_OPTIMIZER_H__
#define __GRAVISBELL_LIBRARY_NN_OPTIMIZER_H__

#include"Layer/NeuralNetwork/IOptimizer.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	/** �I�v�e�B�}�C�U�[���쐬����.[SGD] */
	iOptimizer_SGD* CreateOptimizer_SGD_CPU(U32 i_parameterCount);
	iOptimizer_SGD* CreateOptimizer_SGD_GPU(U32 i_parameterCount);

	/** �I�v�e�B�}�C�U�[���X�V����.�قȂ�^�������ꍇ�͋����I�Ɏw��̌^�ɕϊ������. */
	ErrorCode UpdateOptimizer_SGD_CPU(IOptimizer** io_ppOptimizer, U32 i_parameterCount, F32 i_learnCoeff);
	ErrorCode UpdateOptimizer_SGD_GPU(IOptimizer** io_ppOptimizer, U32 i_parameterCount, F32 i_learnCoeff);


	/** �I�v�e�B�}�C�U�[���쐬����.[Momentum] */
	iOptimizer_Momentum* CreateOptimizer_Momentum_CPU(U32 i_parameterCount);
	iOptimizer_Momentum* CreateOptimizer_Momentum_GPU(U32 i_parameterCount);

	/** �I�v�e�B�}�C�U�[���X�V����.�قȂ�^�������ꍇ�͋����I�Ɏw��̌^�ɕϊ������. */
	ErrorCode UpdateOptimizer_Momentum_CPU(IOptimizer** io_ppOptimizer, U32 i_parameterCount, F32 i_learnCoeff, F32 i_alpha);
	ErrorCode UpdateOptimizer_Momentum_GPU(IOptimizer** io_ppOptimizer, U32 i_parameterCount, F32 i_learnCoeff, F32 i_alpha);



	/** �I�v�e�B�}�C�U�[���쐬����.[AdaDelta] */
	iOptimizer_AdaDelta* CreateOptimizer_AdaDelta_CPU(U32 i_parameterCount);
	iOptimizer_AdaDelta* CreateOptimizer_AdaDelta_GPU(U32 i_parameterCount);

	/** �I�v�e�B�}�C�U�[���쐬����.[Adam] */
	iOptimizer_Adam* CreateOptimizer_Adam_CPU(U32 i_parameterCount);
	iOptimizer_Adam* CreateOptimizer_Adam_GPU(U32 i_parameterCount);

}	// NeuralNetwork
}	// Layer
}	// Gravisbell


#endif
