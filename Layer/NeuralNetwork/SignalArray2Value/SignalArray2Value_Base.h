//======================================
// 出力信号分割レイヤー
//======================================
#ifndef __SignalArray2Value_BASE_H__
#define __SignalArray2Value_BASE_H__

#include<Layer/NeuralNetwork/INNSingle2MultLayer.h>

#include<vector>

#include"SignalArray2Value_DATA.hpp"

#include"SignalArray2Value_LayerData_Base.h"
#include"../_LayerBase/CLayerBase.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	typedef float NEURON_TYPE;	/**< ニューロンに使用するデータ型. float or double */

	class SignalArray2Value_Base : public CNNSingle2SingleLayerBase<>
	{
	public:
		/** コンストラクタ */
		SignalArray2Value_Base(Gravisbell::GUID guid, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct);

		/** デストラクタ */
		virtual ~SignalArray2Value_Base();

	};

}	// NeuralNetwork
}	// Layer
}	// Gravisbell


#endif
