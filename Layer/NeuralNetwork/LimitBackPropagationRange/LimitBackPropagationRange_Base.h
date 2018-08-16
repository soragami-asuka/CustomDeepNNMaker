//======================================
// 出力信号分割レイヤー
//======================================
#ifndef __LimitBackPropagationRange_BASE_H__
#define __LimitBackPropagationRange_BASE_H__

#include<Layer/NeuralNetwork/INNSingle2MultLayer.h>

#include<vector>

#include"LimitBackPropagationRange_DATA.hpp"

#include"LimitBackPropagationRange_LayerData_Base.h"
#include"../_LayerBase/CLayerBase.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	typedef float NEURON_TYPE;	/**< ニューロンに使用するデータ型. float or double */

	class LimitBackPropagationRange_Base : public CNNSingle2SingleLayerBase<>
	{
	public:
		/** コンストラクタ */
		LimitBackPropagationRange_Base(Gravisbell::GUID guid, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct);

		/** デストラクタ */
		virtual ~LimitBackPropagationRange_Base();

	};

}	// NeuralNetwork
}	// Layer
}	// Gravisbell


#endif
