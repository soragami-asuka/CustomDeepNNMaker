//======================================
// �o�b�`���K�����C���[
//======================================
#ifndef __Normalization_Scale_BASE_H__
#define __Normalization_Scale_BASE_H__

#pragma warning(push)
#pragma warning(disable : 4267)
#pragma warning(disable : 4819)
#include <cuda.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "device_launch_parameters.h"
#pragma warning(pop)

#include<vector>
#include<Layer/NeuralNetwork/INNSingle2SingleLayer.h>

#include"Normalization_Scale_DATA.hpp"

#include"Normalization_Scale_LayerData_Base.h"
#include"../_LayerBase/CLayerBase.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	typedef float NEURON_TYPE;	/**< �j���[�����Ɏg�p����f�[�^�^. float or double */

	class Normalization_Scale_Base : public CNNSingle2SingleLayerBase<>
	{
	protected:
	public:
		/** �R���X�g���N�^ */
		Normalization_Scale_Base(Gravisbell::GUID guid, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct);

		/** �f�X�g���N�^ */
		virtual ~Normalization_Scale_Base();

		//===========================
		// ���C���[����
		//===========================
	public:
	};

}	// NeuralNetwork
}	// Layer
}	// Gravisbell


#endif
