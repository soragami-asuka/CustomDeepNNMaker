//=======================================
// ���o�͐M���f�[�^���C���[
//=======================================
#ifndef __GRAVISBELL_I_IO_DATA_LAYER_H__
#define __GRAVISBELL_I_IO_DATA_LAYER_H__

#include"../../Common/Common.h"
#include"../../Common/ErrorCode.h"

#include"../IO/ISingleOutputLayer.h"
#include"../IO/ISingleInputLayer.h"
#include"IDataLayer.h"


namespace Gravisbell {
namespace Layer {
namespace IOData {

	/** ���o�̓f�[�^���C���[ */
	class IIODataLayer : public IO::ISingleOutputLayer, public IO::ISingleInputLayer, public IDataLayer
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

}	// IOData
}	// Layer
}	// Gravisbell

#endif