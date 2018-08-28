//======================================
// �o�͐M���������C���[
//======================================
#ifndef __LimitBackPropagationBox_BASE_H__
#define __LimitBackPropagationBox_BASE_H__

#include<Layer/NeuralNetwork/INNSingle2MultLayer.h>

#include<vector>

#include"LimitBackPropagationBox_DATA.hpp"

#include"LimitBackPropagationBox_LayerData_Base.h"
#include"../_LayerBase/CLayerBase.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	typedef float NEURON_TYPE;	/**< �j���[�����Ɏg�p����f�[�^�^. float or double */

	class LimitBackPropagationBox_Base : public CNNSingle2SingleLayerBase<>
	{
	public:
		/** �R���X�g���N�^ */
		LimitBackPropagationBox_Base(Gravisbell::GUID guid, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct);

		/** �f�X�g���N�^ */
		virtual ~LimitBackPropagationBox_Base();

	};

}	// NeuralNetwork
}	// Layer
}	// Gravisbell


#endif
