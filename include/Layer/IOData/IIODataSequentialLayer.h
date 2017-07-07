//=======================================
// ���o�͐M���f�[�^���C���[(��������)
//=======================================
#ifndef __GRAVISBELL_I_IO_DATA_SEQUENTIAL_LAYER_H__
#define __GRAVISBELL_I_IO_DATA_SEQUENTIAL_LAYER_H__

#include"../../Common/Common.h"
#include"../../Common/ErrorCode.h"

#include"IIODataLayer_base.h"


namespace Gravisbell {
namespace Layer {
namespace IOData {

	/** ���o�̓f�[�^���C���[ */
	class IIODataSequentialLayer : public IIODataLayer_base
	{
	public:
		/** �R���X�g���N�^ */
		IIODataSequentialLayer(){}
		/** �f�X�g���N�^ */
		virtual ~IIODataSequentialLayer(){}

	public:
		/** �f�[�^��ǉ�����.
			@param	lpData	�f�[�^��g�̔z��. [GetBufferSize()�̖߂�l]�̗v�f�����K�v. */
		virtual ErrorCode SetData(U32 dataNo, const float lpData[]) = 0;
		/** �f�[�^��ǉ�����.
			@param	lpData	�f�[�^��g�̔z��. [GetBufferSize()�̖߂�l]�̗v�f�����K�v. 0�`255�̒l. �����I�ɂ�0.0�`1.0�ɕϊ������. */
		virtual ErrorCode SetData(U32 dataNo, const BYTE lpData[]) = 0;
	};

}	// IOData
}	// Layer
}	// Gravisbell

#endif