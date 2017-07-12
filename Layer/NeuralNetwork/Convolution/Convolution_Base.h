//======================================
// �t�B�[�h�t�H���[�h�j���[�����l�b�g���[�N�̓����������C���[
// �����A����������������
//======================================
#ifndef __CONVOLUTION_BASE_H__
#define __CONVOLUTION_BASE_H__

#include<Layer/NeuralNetwork/INNSingle2SingleLayer.h>

#include<vector>

#include"Convolution_DATA.hpp"

#include"Convolution_LayerData_Base.h"

#include"Layer/NeuralNetwork/IOptimizer.h"
#include"../_LayerBase/CLayerBase.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	typedef float NEURON_TYPE;	/**< �j���[�����Ɏg�p����f�[�^�^. float or double */

	class Convolution_Base : public CNNSingle2SingleLayerBase
	{
	public:
		/** �R���X�g���N�^ */
		Convolution_Base(Gravisbell::GUID guid, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct);

		/** �f�X�g���N�^ */
		virtual ~Convolution_Base();

	};

}	// NeuralNetwork
}	// Layer
}	// Gravisbell


#endif
