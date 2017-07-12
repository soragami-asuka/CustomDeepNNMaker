//======================================
// 活性関レイヤー
//======================================
#ifndef __MergeInput_BASE_H__
#define __MergeInput_BASE_H__

#include<Layer/NeuralNetwork/INNMult2SingleLayer.h>

#include<vector>

#include"MergeInput_DATA.hpp"

#include"MergeInput_LayerData_Base.h"
#include"../_LayerBase/CLayerBase.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	typedef float NEURON_TYPE;	/**< ニューロンに使用するデータ型. float or double */

	class MergeInput_Base : public CNNMult2SingleLayerBase
	{
	public:
		/** コンストラクタ */
		MergeInput_Base(Gravisbell::GUID guid, const std::vector<IODataStruct>& i_lpInputDataStruct, const IODataStruct& i_outputDataStruct);

		/** デストラクタ */
		virtual ~MergeInput_Base();

	};

}	// NeuralNetwork
}	// Layer
}	// Gravisbell


#endif
