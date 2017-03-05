//=======================================
// ���o�͐M���f�[�^���C���[
//=======================================
#ifndef __GRAVISBELL_I_IO_DATA_LAYER_H__
#define __GRAVISBELL_I_IO_DATA_LAYER_H__

#include"Common/Common.h"
#include"Common/ErrorCode.h"

#include"ISingleOutputLayer.h"
#include"ISingleInputLayer.h"
#include"IDataLayer.h"


namespace Gravisbell {
namespace NeuralNetwork {

	/** ���o�̓f�[�^���C���[ */
	class IIODataLayer : public ISingleOutputLayer, public ISingleInputLayer, public IDataLayer
	{
	public:
		/** �R���X�g���N�^ */
		IIODataLayer(){}
		/** �f�X�g���N�^ */
		virtual ~IIODataLayer(){}

	public:
		/** �w�K�덷���v�Z����.
			@param i_lppInputBuffer	���̓f�[�^�o�b�t�@. [GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]�̗v�f�����K�v
			���O�̌v�Z���ʂ��g�p���� */
		virtual ErrorCode CalculateLearnError(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer) = 0;
	};

}	// NeuralNetwork
}	// Gravisbell

#endif