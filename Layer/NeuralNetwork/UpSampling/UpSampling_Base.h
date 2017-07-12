//======================================
// �t�B�[�h�t�H���[�h�j���[�����l�b�g���[�N�̓����������C���[
// �����A����������������
//======================================
#ifndef __UpSampling_BASE_H__
#define __UpSampling_BASE_H__

#include<Layer/NeuralNetwork/INNSingle2SingleLayer.h>

#include<vector>

#include"UpSampling_DATA.hpp"

#include"UpSampling_LayerData_Base.h"
#include"../_LayerBase/CLayerBase.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	typedef float NEURON_TYPE;	/**< �j���[�����Ɏg�p����f�[�^�^. float or double */

	class UpSampling_Base : public CNNSingle2SingleLayerBase
	{
	public:
		/** �R���X�g���N�^ */
		UpSampling_Base(Gravisbell::GUID guid, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct);

		/** �f�X�g���N�^ */
		virtual ~UpSampling_Base();

	};

}	// NeuralNetwork
}	// Layer
}	// Gravisbell


#endif
