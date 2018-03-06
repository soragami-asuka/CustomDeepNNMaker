//======================================
// 出力信号分割レイヤー
//======================================
#ifndef __CHOOSEBOX_BASE_H__
#define __CHOOSEBOX_BASE_H__

#include<Layer/NeuralNetwork/INNSingle2MultLayer.h>

#include<vector>

#include"ChooseBox_DATA.hpp"

#include"ChooseBox_LayerData_Base.h"
#include"../_LayerBase/CLayerBase.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	typedef float NEURON_TYPE;	/**< ニューロンに使用するデータ型. float or double */

	class ChooseBox_Base : public CNNSingle2SingleLayerBase<>
	{
	public:
		/** コンストラクタ */
		ChooseBox_Base(Gravisbell::GUID guid, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct);

		/** デストラクタ */
		virtual ~ChooseBox_Base();

	};

}	// NeuralNetwork
}	// Layer
}	// Gravisbell


#endif
