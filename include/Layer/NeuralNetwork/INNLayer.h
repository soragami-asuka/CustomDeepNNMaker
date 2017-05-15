//=======================================
// ���C���[�x�[�X
//=======================================
#ifndef __GRAVISBELL_I_NN_LAYER_BASE_H__
#define __GRAVISBELL_I_NN_LAYER_BASE_H__

#include"../IO/ISingleInputLayer.h"
#include"../IO/ISingleOutputLayer.h"

#include"../../SettingData/Standard/IData.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	/** ���C���[�x�[�X */
	class INNLayer : public IO::ISingleInputLayer, public virtual IO::ISingleOutputLayer
	{
	public:
		/** �R���X�g���N�^ */
		INNLayer(){}
		/** �f�X�g���N�^ */
		virtual ~INNLayer(){}

	public:


	public:
		/** ���Z���������s����.
			@param i_lppInputBuffer	���̓f�[�^�o�b�t�@. [GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]�̗v�f�����K�v
			@return ���������ꍇ0���Ԃ� */
		virtual ErrorCode Calculate(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer) = 0;

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