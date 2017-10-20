//======================================
// 出力信号分割レイヤー
//======================================
#ifndef __CHOOSECHANNEL_BASE_H__
#define __CHOOSECHANNEL_BASE_H__

#include<Layer/NeuralNetwork/INNSingle2MultLayer.h>

#include<vector>

#include"ChooseChannel_DATA.hpp"

#include"ChooseChannel_LayerData_Base.h"
#include"../_LayerBase/CLayerBase.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	typedef float NEURON_TYPE;	/**< ニューロンに使用するデータ型. float or double */

	class ChooseChannel_Base : public CNNSingle2SingleLayerBase<>
	{
	public:
		/** コンストラクタ */
		ChooseChannel_Base(Gravisbell::GUID guid, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct);

		/** デストラクタ */
		virtual ~ChooseChannel_Base();

	};

}	// NeuralNetwork
}	// Layer
}	// Gravisbell


#endif
