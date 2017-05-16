//=======================================
// �������͂�����NN�̃��C���[
//=======================================
#ifndef __GRAVISBELL_I_NN_MULT_INPUT_LAYER_H__
#define __GRAVISBELL_I_NN_MULT_INPUT_LAYER_H__

#include"../IO/IMultInputLayer.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	/** ���C���[�x�[�X */
	class INNMultInputLayer : public IO::IMultInputLayer
	{
	public:
		/** �R���X�g���N�^ */
		INNMultInputLayer() : IMultInputLayer(){}
		/** �f�X�g���N�^ */
		virtual ~INNMultInputLayer(){}

	public:
		/** ���Z���������s����.
			@param i_lppInputBuffer	���̓f�[�^�o�b�t�@. [GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]�̗v�f�����K�v
			@return ���������ꍇ0���Ԃ� */
		virtual ErrorCode Calculate(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer[]) = 0;
	};


}	// NeuralNetwork
}	// Layer
}	// Gravisbell

#endif