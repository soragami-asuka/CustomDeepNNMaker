//======================================
// �����փ��C���[
//======================================
#ifndef __GlobalAveragePooling_BASE_H__
#define __GlobalAveragePooling_BASE_H__

#include<Layer/NeuralNetwork/INNSingle2SingleLayer.h>

#include<vector>

#include"GlobalAveragePooling_DATA.hpp"

#include"GlobalAveragePooling_LayerData_Base.h"
#include"../_LayerBase/CLayerBase.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	typedef float NEURON_TYPE;	/**< �j���[�����Ɏg�p����f�[�^�^. float or double */

	class GlobalAveragePooling_Base : public CNNSingle2SingleLayerBase<>
	{
	public:
		/** �R���X�g���N�^ */
		GlobalAveragePooling_Base(Gravisbell::GUID guid, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct);

		/** �f�X�g���N�^ */
		virtual ~GlobalAveragePooling_Base();

	};

}	// NeuralNetwork
}	// Layer
}	// Gravisbell


#endif
