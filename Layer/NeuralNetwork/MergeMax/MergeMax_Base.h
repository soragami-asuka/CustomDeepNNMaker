//======================================
// 活性関レイヤー
//======================================
#ifndef __MergeMax_BASE_H__
#define __MergeMax_BASE_H__

#include<Layer/NeuralNetwork/INNMult2SingleLayer.h>

#include<vector>

#include"MergeMax_DATA.hpp"

#include"MergeMax_LayerData_Base.h"
#include"../_LayerBase/CLayerBase.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	typedef float NEURON_TYPE;	/**< ニューロンに使用するデータ型. float or double */

	class MergeMax_Base : public CNNMult2SingleLayerBase<>
	{
	public:
		/** コンストラクタ */
		MergeMax_Base(Gravisbell::GUID guid, const std::vector<IODataStruct>& i_lpInputDataStruct, const IODataStruct& i_outputDataStruct);

		/** デストラクタ */
		virtual ~MergeMax_Base();

	};

}	// NeuralNetwork
}	// Layer
}	// Gravisbell


#endif
