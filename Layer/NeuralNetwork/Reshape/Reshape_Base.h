//======================================
// 出力信号分割レイヤー
//======================================
#ifndef __RESHAPE_BASE_H__
#define __RESHAPE_BASE_H__

#include<Layer/NeuralNetwork/INNSingle2MultLayer.h>

#include<vector>

#include"Reshape_DATA.hpp"

#include"Reshape_LayerData_Base.h"
#include"../_LayerBase/CLayerBase.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	typedef float NEURON_TYPE;	/**< ニューロンに使用するデータ型. float or double */

	class Reshape_Base : public CNNSingle2SingleLayerBase<>
	{
	public:
		/** コンストラクタ */
		Reshape_Base(Gravisbell::GUID guid, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct);

		/** デストラクタ */
		virtual ~Reshape_Base();

	};

}	// NeuralNetwork
}	// Layer
}	// Gravisbell


#endif
