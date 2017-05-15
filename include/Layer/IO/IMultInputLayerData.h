//=======================================
// �������͂������C���[�̃��C���[�f�[�^
//=======================================
#ifndef __GRAVISBELL_I_MULT_INPUT_LAYER_DATA_H__
#define __GRAVISBELL_I_MULT_INPUT_LAYER_DATA_H__

#include"../ILayerData.h"

namespace Gravisbell {
namespace Layer {
namespace IO {

	class IMultInputlayerData : public virtual ILayerData
	{
	public:
		/** �R���X�g���N�^ */
		IMultInputlayerData() : ILayerData(){}
		/** �f�X�g���N�^ */
		virtual ~IMultInputlayerData(){}


		//===========================
		// ���̓��C���[�֘A
		//===========================
	public:
		/** ���̓f�[�^�̐����擾���� */
		virtual U32 GetInputDataCount()const = 0;

		/** ���̓f�[�^�\�����擾����.
			@return	���̓f�[�^�\�� */
		virtual IODataStruct GetInputDataStruct(U32 i_dataNum)const = 0;

		/** ���̓o�b�t�@�����擾����. */
		virtual U32 GetInputBufferCount(U32 i_dataNum)const = 0;

	};

}	// IO
}	// Layer
}	// Gravisbell

#endif