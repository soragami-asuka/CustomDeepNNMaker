//=======================================
// �P����̓��C���[
//=======================================
#ifndef __GRAVISBELL_I_SINGLE_INPUT_LAYER_H__
#define __GRAVISBELL_I_SINGLE_INPUT_LAYER_H__

#include"../../Common/ErrorCode.h"
#include"../../Common/IODataStruct.h"

#include"../ILayerBase.h"
#include"IInputLayer.h"

namespace Gravisbell {
namespace Layer {
namespace IO {

	/** ���C���[�x�[�X */
	class ISingleInputLayer : public virtual IInputLayer
	{
	public:
		/** �R���X�g���N�^ */
		ISingleInputLayer(){}
		/** �f�X�g���N�^ */
		virtual ~ISingleInputLayer(){}

	public:
		/** ���̓f�[�^�\�����擾����.
			@return	���̓f�[�^�\�� */
		virtual IODataStruct GetInputDataStruct()const = 0;

		/** ���̓o�b�t�@�����擾����. byte���ł͖����f�[�^�̐��Ȃ̂Œ��� */
		virtual U32 GetInputBufferCount()const = 0;
	};

}	// IO
}	// Layer
}	// Gravisbell

#endif