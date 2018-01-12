//======================================
// �����փ��C���[
//======================================
#ifndef __MergeAverage_BASE_H__
#define __MergeAverage_BASE_H__

#include<Layer/NeuralNetwork/INNMult2SingleLayer.h>

#include<vector>

#include"MergeAverage_DATA.hpp"

#include"MergeAverage_LayerData_Base.h"
#include"../_LayerBase/CLayerBase.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	typedef float NEURON_TYPE;	/**< �j���[�����Ɏg�p����f�[�^�^. float or double */

	class MergeAverage_Base : public CNNMult2SingleLayerBase<>
	{
	public:
		/** �R���X�g���N�^ */
		MergeAverage_Base(Gravisbell::GUID guid, const std::vector<IODataStruct>& i_lpInputDataStruct, const IODataStruct& i_outputDataStruct);

		/** �f�X�g���N�^ */
		virtual ~MergeAverage_Base();

	};

}	// NeuralNetwork
}	// Layer
}	// Gravisbell


#endif
