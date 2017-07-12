//======================================
// フィードフォワードニューラルネットワークの統合処理レイヤー
// 結合、活性化を処理する
//======================================
#ifndef __CONVOLUTION_BASE_H__
#define __CONVOLUTION_BASE_H__

#include<Layer/NeuralNetwork/INNSingle2SingleLayer.h>

#include<vector>

#include"Convolution_DATA.hpp"

#include"Convolution_LayerData_Base.h"

#include"Layer/NeuralNetwork/IOptimizer.h"
#include"../_LayerBase/CLayerBase.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	typedef float NEURON_TYPE;	/**< ニューロンに使用するデータ型. float or double */

	class Convolution_Base : public CNNSingle2SingleLayerBase
	{
	public:
		/** コンストラクタ */
		Convolution_Base(Gravisbell::GUID guid, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct);

		/** デストラクタ */
		virtual ~Convolution_Base();

	};

}	// NeuralNetwork
}	// Layer
}	// Gravisbell


#endif
