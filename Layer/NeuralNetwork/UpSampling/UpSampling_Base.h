//======================================
// フィードフォワードニューラルネットワークの統合処理レイヤー
// 結合、活性化を処理する
//======================================
#ifndef __UpSampling_BASE_H__
#define __UpSampling_BASE_H__

#include<Layer/NeuralNetwork/INNSingle2SingleLayer.h>

#include<vector>

#include"UpSampling_DATA.hpp"

#include"UpSampling_LayerData_Base.h"
#include"../_LayerBase/CLayerBase.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	typedef float NEURON_TYPE;	/**< ニューロンに使用するデータ型. float or double */

	class UpSampling_Base : public CNNSingle2SingleLayerBase
	{
	public:
		/** コンストラクタ */
		UpSampling_Base(Gravisbell::GUID guid, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct);

		/** デストラクタ */
		virtual ~UpSampling_Base();

	};

}	// NeuralNetwork
}	// Layer
}	// Gravisbell


#endif
