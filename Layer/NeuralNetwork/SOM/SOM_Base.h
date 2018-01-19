//======================================
// フィードフォワードニューラルネットワークの統合処理レイヤー
// 結合、活性化を処理する
//======================================
#ifndef __SOM_BASE_H__
#define __SOM_BASE_H__

#include<vector>
#include<Layer/NeuralNetwork/INNSingle2SingleLayer.h>

#include"SOM_DATA.hpp"

#include"SOM_LayerData_Base.h"
#include"../_LayerBase/CLayerBase.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	typedef float NEURON_TYPE;	/**< ニューロンに使用するデータ型. float or double */

	class SOM_Base : public CNNSingle2SingleLayerBase<SOM::RuntimeParameterStructure>
	{
	public:
		/** コンストラクタ */
		SOM_Base(Gravisbell::GUID guid, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct);

		/** デストラクタ */
		virtual ~SOM_Base();

		//===========================
		// 固有関数
		//===========================
	public:
		/** ユニット数を取得する */
		U32 GetUnitCount()const;

	};

}	// NeuralNetwork
}	// Layer
}	// Gravisbell


#endif
