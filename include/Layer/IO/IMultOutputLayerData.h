//=======================================
// �����o�͂������C���[�̃��C���[�f�[�^
//=======================================
#ifndef __GRAVISBELL_I_MULT_OUTPUT_LAYER_DATA_H__
#define __GRAVISBELL_I_MULT_OUTPUT_LAYER_DATA_H__

#include"../ILayerData.h"

namespace Gravisbell {
namespace Layer {
namespace IO {

	class IMultOutputLayerData : public ILayerData
	{
	public:
		/** �R���X�g���N�^ */
		IMultOutputLayerData() : ILayerData(){}
		/** �f�X�g���N�^ */
		virtual ~IMultOutputLayerData(){}


		//===========================
		// �o�̓��C���[�֘A
		//===========================
	public:
		/** �o�̓f�[�^�̐����擾���� */
		virtual U32 GetOutputDataCount()const = 0;

		/** �o�̓f�[�^�\�����擾���� */
		virtual IODataStruct GetOutputDataStruct(U32 i_dataNum)const = 0;

		/** �o�̓o�b�t�@�����擾���� */
		virtual U32 GetOutputBufferCount(U32 i_dataNum)const = 0;
	};

}	// IO
}	// Layer
}	// Gravisbell

#endif