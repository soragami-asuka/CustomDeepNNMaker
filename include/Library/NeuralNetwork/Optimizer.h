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

	/** �I�v�e�B�}�C�U�[���X�V����.�قȂ�^�������ꍇ�͋����I�Ɏw��̌^�ɕϊ������. */
	ErrorCode UpdateOptimizer_AdaDelta_CPU(IOptimizer** io_ppOptimizer, U32 i_parameterCount, F32 i_beta, F32 i_epsilon);
	ErrorCode UpdateOptimizer_AdaDelta_GPU(IOptimizer** io_ppOptimizer, U32 i_parameterCount, F32 i_beta, F32 i_epsilon);


	/** �I�v�e�B�}�C�U�[���쐬����.[Adam] */
	iOptimizer_Adam* CreateOptimizer_Adam_CPU(U32 i_parameterCount);
	iOptimizer_Adam* CreateOptimizer_Adam_GPU(U32 i_parameterCount);
	
	/** �I�v�e�B�}�C�U�[���X�V����.�قȂ�^�������ꍇ�͋����I�Ɏw��̌^�ɕϊ������. */
	ErrorCode UpdateOptimizer_Adam_CPU(IOptimizer** io_ppOptimizer, U32 i_parameterCount, F32 i_alpha, F32 i_beta1, F32 i_beta2, F32 i_epsilon);
	ErrorCode UpdateOptimizer_Adam_GPU(IOptimizer** io_ppOptimizer, U32 i_parameterCount, F32 i_alpha, F32 i_beta1, F32 i_beta2, F32 i_epsilon);

}	// NeuralNetwork
}	// Layer
}	// Gravisbell


#endif
