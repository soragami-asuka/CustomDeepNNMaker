//======================================
// �o�͐M���������C���[
//======================================
#ifndef __ProbabilityArray2Value_BASE_H__
#define __ProbabilityArray2Value_BASE_H__

#include<Layer/NeuralNetwork/INNSingle2MultLayer.h>

#include<vector>

#include"ProbabilityArray2Value_DATA.hpp"

#include"ProbabilityArray2Value_LayerData_Base.h"
#include"../_LayerBase/CLayerBase.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	typedef float NEURON_TYPE;	/**< �j���[�����Ɏg�p����f�[�^�^. float or double */

	class ProbabilityArray2Value_Base : public CNNSingle2SingleLayerBase<>
	{
	public:
		/** �R���X�g���N�^ */
		ProbabilityArray2Value_Base(Gravisbell::GUID guid, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct);

		/** �f�X�g���N�^ */
		virtual ~ProbabilityArray2Value_Base();

	};

}	// NeuralNetwork
}	// Layer
}	// Gravisbell


#endif
