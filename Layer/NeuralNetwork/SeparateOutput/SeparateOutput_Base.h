//======================================
// �o�͐M���������C���[
//======================================
#ifndef __SEPARATEOUTPUT_BASE_H__
#define __SEPARATEOUTPUT_BASE_H__

#include<Layer/NeuralNetwork/INNSingle2MultLayer.h>

#include<vector>

#include"SeparateOutput_DATA.hpp"

#include"SeparateOutput_LayerData_Base.h"
#include"../_LayerBase/CLayerBase.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	typedef float NEURON_TYPE;	/**< �j���[�����Ɏg�p����f�[�^�^. float or double */

	class SeparateOutput_Base : public CNNSingle2MultLayerBase<>
	{
	public:
		/** �R���X�g���N�^ */
		SeparateOutput_Base(Gravisbell::GUID guid, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct);

		/** �f�X�g���N�^ */
		virtual ~SeparateOutput_Base();

		//===========================
		// �o�̓��C���[�֘A
		//===========================
	public:
		/** �o�̓f�[�^�̏o�͐惌�C���[��. */
		U32 GetOutputToLayerCount()const;
	};

}	// NeuralNetwork
}	// Layer
}	// Gravisbell


#endif
