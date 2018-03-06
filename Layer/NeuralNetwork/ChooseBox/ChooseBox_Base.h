//======================================
// �o�͐M���������C���[
//======================================
#ifndef __CHOOSEBOX_BASE_H__
#define __CHOOSEBOX_BASE_H__

#include<Layer/NeuralNetwork/INNSingle2MultLayer.h>

#include<vector>

#include"ChooseBox_DATA.hpp"

#include"ChooseBox_LayerData_Base.h"
#include"../_LayerBase/CLayerBase.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	typedef float NEURON_TYPE;	/**< �j���[�����Ɏg�p����f�[�^�^. float or double */

	class ChooseBox_Base : public CNNSingle2SingleLayerBase<>
	{
	public:
		/** �R���X�g���N�^ */
		ChooseBox_Base(Gravisbell::GUID guid, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct);

		/** �f�X�g���N�^ */
		virtual ~ChooseBox_Base();

	};

}	// NeuralNetwork
}	// Layer
}	// Gravisbell


#endif
