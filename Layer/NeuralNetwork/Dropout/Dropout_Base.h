//======================================
// �����փ��C���[
//======================================
#ifndef __Dropout_BASE_H__
#define __Dropout_BASE_H__

#include<Layer/NeuralNetwork/INNSingle2SingleLayer.h>

#include<vector>

#include"Dropout_DATA.hpp"

#include"Dropout_LayerData_Base.h"
#include"../_LayerBase/CLayerBase.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	typedef float NEURON_TYPE;	/**< �j���[�����Ɏg�p����f�[�^�^. float or double */

	class Dropout_Base : public CNNSingle2SingleLayerBase<Dropout::RuntimeParameterStructure>
	{
	public:
		/** �R���X�g���N�^ */
		Dropout_Base(Gravisbell::GUID guid, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct);

		/** �f�X�g���N�^ */
		virtual ~Dropout_Base();

	};

}	// NeuralNetwork
}	// Layer
}	// Gravisbell


#endif
