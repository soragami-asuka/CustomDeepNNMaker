//======================================
// �o�͐M���������C���[
//======================================
#ifndef __LimitBackPropagationRange_BASE_H__
#define __LimitBackPropagationRange_BASE_H__

#include<Layer/NeuralNetwork/INNSingle2MultLayer.h>

#include<vector>

#include"LimitBackPropagationRange_DATA.hpp"

#include"LimitBackPropagationRange_LayerData_Base.h"
#include"../_LayerBase/CLayerBase.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	typedef float NEURON_TYPE;	/**< �j���[�����Ɏg�p����f�[�^�^. float or double */

	class LimitBackPropagationRange_Base : public CNNSingle2SingleLayerBase<>
	{
	public:
		/** �R���X�g���N�^ */
		LimitBackPropagationRange_Base(Gravisbell::GUID guid, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct);

		/** �f�X�g���N�^ */
		virtual ~LimitBackPropagationRange_Base();

	};

}	// NeuralNetwork
}	// Layer
}	// Gravisbell


#endif
