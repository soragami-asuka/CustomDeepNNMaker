//======================================
// �t�B�[�h�t�H���[�h�j���[�����l�b�g���[�N�̓����������C���[
// �����A����������������
//======================================
#ifndef __FullyConnect_BASE_H__
#define __FullyConnect_BASE_H__

#include<vector>
#include<Layer/NeuralNetwork/INNSingle2SingleLayer.h>

#include"FullyConnect_DATA.hpp"

#include"FullyConnect_LayerData_Base.h"
#include"../_LayerBase/CLayerBase.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	typedef float NEURON_TYPE;	/**< �j���[�����Ɏg�p����f�[�^�^. float or double */

	class FullyConnect_Base : public CNNSingle2SingleLayerBase<>
	{
	public:
		/** �R���X�g���N�^ */
		FullyConnect_Base(Gravisbell::GUID guid, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct);

		/** �f�X�g���N�^ */
		virtual ~FullyConnect_Base();

		//===========================
		// �ŗL�֐�
		//===========================
	public:
		/** �j���[���������擾���� */
		U32 GetNeuronCount()const;

	};

}	// NeuralNetwork
}	// Layer
}	// Gravisbell


#endif
