//=======================================
// �P��o�͂�����NN�̃��C���[
//=======================================
#ifndef __GRAVISBELL_I_NN_SINGLE_OUTPUT_LAYER_H__
#define __GRAVISBELL_I_NN_SINGLE_OUTPUT_LAYER_H__

#include"../IO/ISingleOutputLayer.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	/** ���C���[�x�[�X */
	class INNSingleOutputLayer : public IO::ISingleOutputLayer
	{
	public:
		/** �R���X�g���N�^ */
		INNSingleOutputLayer(){}
		/** �f�X�g���N�^ */
		virtual ~INNSingleOutputLayer(){}


	public:
		/** �w�K���������s����.
			���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
			@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v.
			���O�̌v�Z���ʂ��g�p���� */
		virtual ErrorCode Training(CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer) = 0;
	};


}	// NeuralNetwork
}	// Layer
}	// Gravisbell

#endif