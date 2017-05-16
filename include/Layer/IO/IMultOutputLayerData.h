//=======================================
// �����o�͂������C���[�̃��C���[�f�[�^
//=======================================
#ifndef __GRAVISBELL_I_MULT_OUTPUT_LAYER_DATA_H__
#define __GRAVISBELL_I_MULT_OUTPUT_LAYER_DATA_H__

#include"../ILayerData.h"

namespace Gravisbell {
namespace Layer {
namespace IO {

	class IMultOutputLayerData : public virtual ILayerData
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
		/** �o�̓f�[�^�̏o�͐惌�C���[��. */
		virtual U32 GetOutputToLayerCount()const = 0;

		/** �o�̓f�[�^�\�����擾���� */
		virtual IODataStruct GetOutputDataStruct()const = 0;

		/** �o�̓o�b�t�@�����擾���� */
		virtual U32 GetOutputBufferCount()const = 0;
	};

}	// IO
}	// Layer
}	// Gravisbell

#endif