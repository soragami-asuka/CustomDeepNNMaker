//======================================
// 活性関レイヤー
//======================================
#ifndef __Dropout_BASE_H__
#define __Dropout_BASE_H__

#include<Layer/NeuralNetwork/INNSingle2SingleLayer.h>

#include<vector>

#include"Dropout_DATA.hpp"

#include"Dropout_LayerData_Base.h"
#include"../_LayerBase/CLayerBase.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	typedef float NEURON_TYPE;	/**< ニューロンに使用するデータ型. float or double */

	class Dropout_Base : public CNNSingle2SingleLayerBase<Dropout::RuntimeParameterStructure>
	{
	public:
		/** コンストラクタ */
		Dropout_Base(Gravisbell::GUID guid, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct);

		/** デストラクタ */
		virtual ~Dropout_Base();

	};

}	// NeuralNetwork
}	// Layer
}	// Gravisbell


#endif
