//======================================
// 活性関レイヤー
//======================================
#ifndef __Residual_BASE_H__
#define __Residual_BASE_H__

#include<Layer/NeuralNetwork/INNMult2SingleLayer.h>

#include<vector>

#include"Residual_DATA.hpp"

#include"Residual_LayerData_Base.h"
#include"../_LayerBase/CLayerBase.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	typedef float NEURON_TYPE;	/**< ニューロンに使用するデータ型. float or double */

	class Residual_Base : public CNNMult2SingleLayerBase
	{
	public:
		/** コンストラクタ */
		Residual_Base(Gravisbell::GUID guid, const std::vector<IODataStruct>& i_lpInputDataStruct, const IODataStruct& i_outputDataStruct);

		/** デストラクタ */
		virtual ~Residual_Base();

	};

}	// NeuralNetwork
}	// Layer
}	// Gravisbell


#endif
