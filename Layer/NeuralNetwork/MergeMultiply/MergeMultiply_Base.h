//======================================
// 活性関レイヤー
//======================================
#ifndef __MergeMultiply_BASE_H__
#define __MergeMultiply_BASE_H__

#include<Layer/NeuralNetwork/INNMult2SingleLayer.h>

#include<vector>

#include"MergeMultiply_DATA.hpp"

#include"MergeMultiply_LayerData_Base.h"
#include"../_LayerBase/CLayerBase.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	typedef float NEURON_TYPE;	/**< ニューロンに使用するデータ型. float or double */

	class MergeMultiply_Base : public CNNMult2SingleLayerBase<>
	{
	public:
		/** コンストラクタ */
		MergeMultiply_Base(Gravisbell::GUID guid, const std::vector<IODataStruct>& i_lpInputDataStruct, const IODataStruct& i_outputDataStruct);

		/** デストラクタ */
		virtual ~MergeMultiply_Base();

	};

}	// NeuralNetwork
}	// Layer
}	// Gravisbell


#endif
