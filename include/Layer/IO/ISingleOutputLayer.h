//=======================================
// �P��o�̓��C���[
//=======================================
#ifndef __GRAVISBELL_I_SINGLE_OUTPUT_LAYER_H__
#define __GRAVISBELL_I_SINGLE_OUTPUT_LAYER_H__

#include"../../Common/ErrorCode.h"
#include"../../Common/IODataStruct.h"

#include"../ILayerBase.h"
#include"IOutputLayer.h"

namespace Gravisbell {
namespace Layer {
namespace IO {

	/** ���C���[�x�[�X */
	class ISingleOutputLayer : public virtual IOutputLayer
	{
	public:
		/** �R���X�g���N�^ */
		ISingleOutputLayer(){}
		/** �f�X�g���N�^ */
		virtual ~ISingleOutputLayer(){}

	public:
		/** �o�̓f�[�^�\�����擾����.
			@return	�o�̓f�[�^�\�� */
		virtual IODataStruct GetOutputDataStruct()const = 0;

		/** �o�̓o�b�t�@�����擾����. byte���ł͖����f�[�^�̐��Ȃ̂Œ��� */
		virtual U32 GetOutputBufferCount()const = 0;
	};

}	// IO
}	// Layer
}	// GravisBell

#endif