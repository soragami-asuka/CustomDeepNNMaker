//======================================
// �����փ��C���[
//======================================
#ifndef __ACTIVATION_BASE_H__
#define __ACTIVATION_BASE_H__

#include<Layer/NeuralNetwork/INNSingle2SingleLayer.h>

#include<vector>

#include"Activation_DATA.hpp"

#include"Activation_LayerData_Base.h"
#include"../_LayerBase/CLayerBase.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	typedef float NEURON_TYPE;	/**< �j���[�����Ɏg�p����f�[�^�^. float or double */

	class Activation_Base : public CNNSingle2SingleLayerBase
	{
	protected:
	public:
		/** �R���X�g���N�^ */
		Activation_Base(Gravisbell::GUID guid, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct);

		/** �f�X�g���N�^ */
		virtual ~Activation_Base();

		//===========================
		// ���C���[����
		//===========================
	public:




		//===========================
		// �ŗL�֐�
		//===========================
	public:


	};

}	// NeuralNetwork
}	// Layer
}	// Gravisbell


#endif
