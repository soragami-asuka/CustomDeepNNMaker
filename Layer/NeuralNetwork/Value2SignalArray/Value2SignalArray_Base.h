//======================================
// �o�͐M���������C���[
//======================================
#ifndef __Value2SignalArray_BASE_H__
#define __Value2SignalArray_BASE_H__

#include<Layer/NeuralNetwork/INNSingle2MultLayer.h>

#include<vector>

#include"Value2SignalArray_DATA.hpp"

#include"Value2SignalArray_LayerData_Base.h"
#include"../_LayerBase/CLayerBase.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	typedef float NEURON_TYPE;	/**< �j���[�����Ɏg�p����f�[�^�^. float or double */

	class Value2SignalArray_Base : public CNNSingle2SingleLayerBase<>
	{
	public:
		/** �R���X�g���N�^ */
		Value2SignalArray_Base(Gravisbell::GUID guid, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct);

		/** �f�X�g���N�^ */
		virtual ~Value2SignalArray_Base();

	};

}	// NeuralNetwork
}	// Layer
}	// Gravisbell


#endif
