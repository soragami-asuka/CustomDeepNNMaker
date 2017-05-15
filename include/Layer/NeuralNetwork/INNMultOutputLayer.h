//=======================================
// �����o�͂�����NN�̃��C���[
//=======================================
#ifndef __GRAVISBELL_I_NN_MULT_OUTPUT_LAYER_H__
#define __GRAVISBELL_I_NN_MULT_OUTPUT_LAYER_H__

#include"../IO/IMultOutputLayer.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	/** ���C���[�x�[�X */
	class INNMultOutputLayer : public IO::IMultOutputLayer
	{
	public:
		/** �R���X�g���N�^ */
		INNMultOutputLayer(){}
		/** �f�X�g���N�^ */
		virtual ~INNMultOutputLayer(){}


	public:
		/** �w�K���������s����.
			���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
			@param	i_lppDOutputBuffer[]	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v�Ȕz���[GetOutputDataCount()]�z��
			���O�̌v�Z���ʂ��g�p���� */
		virtual ErrorCode Training(CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer[]) = 0;
	};


}	// NeuralNetwork
}	// Layer
}	// Gravisbell

#endif