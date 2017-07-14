//======================================
// 活性関レイヤー
//======================================
#ifndef __POOLING_BASE_H__
#define __POOLING_BASE_H__

#include<Layer/NeuralNetwork/INNSingle2SingleLayer.h>

#include<vector>

#include"Pooling_DATA.hpp"

#include"Pooling_LayerData_Base.h"
#include"../_LayerBase/CLayerBase.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	typedef float NEURON_TYPE;	/**< ニューロンに使用するデータ型. float or double */

	class Pooling_Base : public CNNSingle2SingleLayerBase<>
	{
	public:
		/** コンストラクタ */
		Pooling_Base(Gravisbell::GUID guid, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct);

		/** デストラクタ */
		virtual ~Pooling_Base();
	};

}	// NeuralNetwork
}	// Layer
}	// Gravisbell


#endif
