//======================================
// �����փ��C���[
//======================================
#ifndef __GAUSSIANNOISE_BASE_H__
#define __GAUSSIANNOISE_BASE_H__

#include<Layer/NeuralNetwork/INNSingle2SingleLayer.h>

#include<vector>

#include"GaussianNoise_DATA.hpp"

#include"GaussianNoise_LayerData_Base.h"
#include"../_LayerBase/CLayerBase.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	typedef float NEURON_TYPE;	/**< �j���[�����Ɏg�p����f�[�^�^. float or double */

	class GaussianNoise_Base : public CNNSingle2SingleLayerBase<GaussianNoise::RuntimeParameterStructure>
	{
	public:
		/** �R���X�g���N�^ */
		GaussianNoise_Base(Gravisbell::GUID guid, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct);

		/** �f�X�g���N�^ */
		virtual ~GaussianNoise_Base();
	};

}	// NeuralNetwork
}	// Layer
}	// Gravisbell


#endif
