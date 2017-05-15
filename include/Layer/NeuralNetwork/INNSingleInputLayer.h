//=======================================
// �P����͂�����NN�̃��C���[
//=======================================
#ifndef __GRAVISBELL_I_NN_SINGLE_INPUT_LAYER_H__
#define __GRAVISBELL_I_NN_SINGLE_INPUT_LAYER_H__

#include"../IO/ISingleInputLayer.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	/** ���C���[�x�[�X */
	class INNSingleInputLayer : public IO::ISingleInputLayer
	{
	public:
		/** �R���X�g���N�^ */
		INNSingleInputLayer() : ISingleInputLayer(){}
		/** �f�X�g���N�^ */
		virtual ~INNSingleInputLayer(){}

	public:
		/** ���Z���������s����.
			@param i_lppInputBuffer	���̓f�[�^�o�b�t�@. [GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]�̗v�f�����K�v
			@return ���������ꍇ0���Ԃ� */
		virtual ErrorCode Calculate(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer) = 0;
	};


}	// NeuralNetwork
}	// Layer
}	// Gravisbell

#endif