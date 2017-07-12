//======================================
// 活性関レイヤー
//======================================
#ifndef __Activation_Discriminator_BASE_H__
#define __Activation_Discriminator_BASE_H__

#include<Layer/NeuralNetwork/INNSingle2SingleLayer.h>

#include<vector>

#include"Activation_Discriminator_DATA.hpp"

#include"Activation_Discriminator_LayerData_Base.h"
#include"../_LayerBase/CLayerBase.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class Activation_Discriminator_Base : public CNNSingle2SingleLayerBase
	{
	public:
		/** コンストラクタ */
		Activation_Discriminator_Base(Gravisbell::GUID guid, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct);

		/** デストラクタ */
		virtual ~Activation_Discriminator_Base();

	};

}	// NeuralNetwork
}	// Layer
}	// Gravisbell


#endif
