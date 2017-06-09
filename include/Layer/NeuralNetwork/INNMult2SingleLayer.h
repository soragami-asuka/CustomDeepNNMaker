//=======================================
// �������́A�P��o�͂�����NN�̃��C���[
//=======================================
#ifndef __GRAVISBELL_I_NN_MULT_TO_SINGLE_LAYER_H__
#define __GRAVISBELL_I_NN_MULT_TO_SINGLE_LAYER_H__

#include"../IO/IMultInputLayer.h"
#include"../IO/ISingleOutputLayer.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	/** ���C���[�x�[�X */
	class INNMult2SingleLayer : public IO::IMultInputLayer, public IO::ISingleOutputLayer
	{
	public:
		/** �R���X�g���N�^ */
		INNMult2SingleLayer() : IMultInputLayer(), ISingleOutputLayer(){}
		/** �f�X�g���N�^ */
		virtual ~INNMult2SingleLayer(){}

	public:
		/** ���Z���������s����.
			@param i_lppInputBuffer	���̓f�[�^�o�b�t�@. [GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]�̗v�f�����K�v
			@return ���������ꍇ0���Ԃ� */
		virtual ErrorCode Calculate(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer[]) = 0;

		/** �w�K���������s����.
			���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
			@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v.
			���O�̌v�Z���ʂ��g�p���� */
		virtual ErrorCode Training(BATCH_BUFFER_POINTER o_lppDInputBuffer[], CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer) = 0;
	};


}	// NeuralNetwork
}	// Layer
}	// Gravisbell

#endif