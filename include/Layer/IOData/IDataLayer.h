//=======================================
// �f�[�^���C���[
//=======================================
#ifndef __GRAVISBELL_I_DATA_LAYER_H__
#define __GRAVISBELL_I_DATA_LAYER_H__

#include"../../Common/ErrorCode.h"
#include"../../Common/IODataStruct.h"

namespace Gravisbell {
namespace Layer {
namespace IOData {


	/** ���̓f�[�^���C���[ */
	class IDataLayer
	{
	public:
		/** �R���X�g���N�^ */
		IDataLayer(){}
		/** �f�X�g���N�^ */
		virtual ~IDataLayer(){}

	public:
		/** �f�[�^�̍\�������擾���� */
		virtual IODataStruct GetDataStruct()const = 0;

		/** �f�[�^�̃o�b�t�@�T�C�Y���擾����.
			@return �f�[�^�̃o�b�t�@�T�C�Y.�g�p����float�^�z��̗v�f��. */
		virtual U32 GetBufferCount()const = 0;

		/** �f�[�^�����擾���� */
		virtual U32 GetDataCount()const = 0;

	};

}	// IOData
}	// Layer
}	// Gravisbell

#endif