//======================================
// �t�B�[�h�t�H���[�h�j���[�����l�b�g���[�N�̓����������C���[
// �����A����������������
//======================================
#ifndef __SOM_BASE_H__
#define __SOM_BASE_H__

#include<vector>
#include<Layer/NeuralNetwork/INNSingle2SingleLayer.h>

#include"SOM_DATA.hpp"

#include"SOM_LayerData_Base.h"
#include"../_LayerBase/CLayerBase.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	typedef float NEURON_TYPE;	/**< �j���[�����Ɏg�p����f�[�^�^. float or double */

	class SOM_Base : public CNNSingle2SingleLayerBase<SOM::RuntimeParameterStructure>
	{
	public:
		/** �R���X�g���N�^ */
		SOM_Base(Gravisbell::GUID guid, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct);

		/** �f�X�g���N�^ */
		virtual ~SOM_Base();

		//===========================
		// �ŗL�֐�
		//===========================
	public:
		/** ���j�b�g�����擾���� */
		U32 GetUnitCount()const;

	};

}	// NeuralNetwork
}	// Layer
}	// Gravisbell


#endif
