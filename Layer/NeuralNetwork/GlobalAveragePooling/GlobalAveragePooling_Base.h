//======================================
// 活性関レイヤー
//======================================
#ifndef __GlobalAveragePooling_BASE_H__
#define __GlobalAveragePooling_BASE_H__

#include<Layer/NeuralNetwork/INNSingle2SingleLayer.h>

#include<vector>

#include"GlobalAveragePooling_DATA.hpp"

#include"GlobalAveragePooling_LayerData_Base.h"
#include"../_LayerBase/CLayerBase.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	typedef float NEURON_TYPE;	/**< ニューロンに使用するデータ型. float or double */

	class GlobalAveragePooling_Base : public CNNSingle2SingleLayerBase<>
	{
	public:
		/** コンストラクタ */
		GlobalAveragePooling_Base(Gravisbell::GUID guid, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct);

		/** デストラクタ */
		virtual ~GlobalAveragePooling_Base();

	};

}	// NeuralNetwork
}	// Layer
}	// Gravisbell


#endif
