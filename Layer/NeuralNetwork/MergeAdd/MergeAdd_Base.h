//======================================
// �����փ��C���[
//======================================
#ifndef __MergeAdd_BASE_H__
#define __MergeAdd_BASE_H__

#include<Layer/NeuralNetwork/INNMult2SingleLayer.h>

#include<vector>

#include"MergeAdd_DATA.hpp"

#include"MergeAdd_LayerData_Base.h"
#include"../_LayerBase/CLayerBase.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	typedef float NEURON_TYPE;	/**< �j���[�����Ɏg�p����f�[�^�^. float or double */

	class MergeAdd_Base : public CNNMult2SingleLayerBase<>
	{
	public:
		/** �R���X�g���N�^ */
		MergeAdd_Base(Gravisbell::GUID guid, const std::vector<IODataStruct>& i_lpInputDataStruct, const IODataStruct& i_outputDataStruct);

		/** �f�X�g���N�^ */
		virtual ~MergeAdd_Base();

	};

}	// NeuralNetwork
}	// Layer
}	// Gravisbell


#endif
