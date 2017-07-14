//======================================
// 出力信号分割レイヤー
//======================================
#ifndef __SEPARATEOUTPUT_BASE_H__
#define __SEPARATEOUTPUT_BASE_H__

#include<Layer/NeuralNetwork/INNSingle2MultLayer.h>

#include<vector>

#include"SeparateOutput_DATA.hpp"

#include"SeparateOutput_LayerData_Base.h"
#include"../_LayerBase/CLayerBase.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	typedef float NEURON_TYPE;	/**< ニューロンに使用するデータ型. float or double */

	class SeparateOutput_Base : public CNNSingle2MultLayerBase<>
	{
	public:
		/** コンストラクタ */
		SeparateOutput_Base(Gravisbell::GUID guid, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct);

		/** デストラクタ */
		virtual ~SeparateOutput_Base();

		//===========================
		// 出力レイヤー関連
		//===========================
	public:
		/** 出力データの出力先レイヤー数. */
		U32 GetOutputToLayerCount()const;
	};

}	// NeuralNetwork
}	// Layer
}	// Gravisbell


#endif
