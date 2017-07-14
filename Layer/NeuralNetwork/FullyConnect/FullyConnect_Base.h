//======================================
// フィードフォワードニューラルネットワークの統合処理レイヤー
// 結合、活性化を処理する
//======================================
#ifndef __FullyConnect_BASE_H__
#define __FullyConnect_BASE_H__

#include<vector>
#include<Layer/NeuralNetwork/INNSingle2SingleLayer.h>

#include"FullyConnect_DATA.hpp"

#include"FullyConnect_LayerData_Base.h"
#include"../_LayerBase/CLayerBase.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	typedef float NEURON_TYPE;	/**< ニューロンに使用するデータ型. float or double */

	class FullyConnect_Base : public CNNSingle2SingleLayerBase<>
	{
	public:
		/** コンストラクタ */
		FullyConnect_Base(Gravisbell::GUID guid, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct);

		/** デストラクタ */
		virtual ~FullyConnect_Base();

		//===========================
		// 固有関数
		//===========================
	public:
		/** ニューロン数を取得する */
		U32 GetNeuronCount()const;

	};

}	// NeuralNetwork
}	// Layer
}	// Gravisbell


#endif
