//=======================================
// �P�Əo�͂������C���[�̃��C���[�f�[�^
//=======================================
#ifndef __GRAVISBELL_I_SINGLE_OUTPUT_LAYER_DATA_H__
#define __GRAVISBELL_I_SINGLE_OUTPUT_LAYER_DATA_H__

#include"../ILayerData.h"

namespace Gravisbell {
namespace Layer {
namespace IO {

	class ISingleOutputLayerData : public virtual ILayerData
	{
	public:
		/** �R���X�g���N�^ */
		ISingleOutputLayerData() : ILayerData(){}
		/** �f�X�g���N�^ */
		virtual ~ISingleOutputLayerData(){}


		//===========================
		// �o�̓��C���[�֘A
		//===========================
	public:
		/** �o�̓f�[�^�\�����擾���� */
		virtual IODataStruct GetOutputDataStruct()const = 0;

		/** �o�̓o�b�t�@�����擾���� */
		virtual U32 GetOutputBufferCount()const = 0;
	};

}	// IO
}	// Layer
}	// Gravisbell

#endif