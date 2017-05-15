//=======================================
// �P�Ɠ��͂������C���[�̃��C���[�f�[�^
//=======================================
#ifndef __GRAVISBELL_I_SINGLE_INPUT_LAYER_DATA_H__
#define __GRAVISBELL_I_SINGLE_INPUT_LAYER_DATA_H__

#include"../ILayerData.h"

namespace Gravisbell {
namespace Layer {
namespace IO {

	class ISingleInputLayerData : public virtual ILayerData
	{
	public:
		/** �R���X�g���N�^ */
		ISingleInputLayerData() : ILayerData(){}
		/** �f�X�g���N�^ */
		virtual ~ISingleInputLayerData(){}


		//===========================
		// ���̓��C���[�֘A
		//===========================
	public:
		/** ���̓f�[�^�\�����擾����.
			@return	���̓f�[�^�\�� */
		virtual IODataStruct GetInputDataStruct()const = 0;

		/** ���̓o�b�t�@�����擾����. */
		virtual U32 GetInputBufferCount()const = 0;

	};

}	// IOD
}	// Layer
}	// Gravisbell

#endif