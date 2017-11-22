//======================================
// �o�͐M���������C���[
//======================================
#ifndef __RESHAPE_SQUARECENTERCROSS_BASE_H__
#define __RESHAPE_SQUARECENTERCROSS_BASE_H__

#include<Layer/NeuralNetwork/INNSingle2MultLayer.h>

#include<vector>

#include"Reshape_SquaresZeroSideLeftTop_DATA.hpp"

#include"Reshape_SquaresZeroSideLeftTop_LayerData_Base.h"
#include"../_LayerBase/CLayerBase.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	typedef float NEURON_TYPE;	/**< �j���[�����Ɏg�p����f�[�^�^. float or double */

	class Reshape_SquaresZeroSideLeftTop_Base : public CNNSingle2SingleLayerBase<>
	{
	public:
		/** �R���X�g���N�^ */
		Reshape_SquaresZeroSideLeftTop_Base(Gravisbell::GUID guid, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct);

		/** �f�X�g���N�^ */
		virtual ~Reshape_SquaresZeroSideLeftTop_Base();

	};

}	// NeuralNetwork
}	// Layer
}	// Gravisbell


#endif
