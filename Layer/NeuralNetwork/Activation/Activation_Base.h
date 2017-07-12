//======================================
// 活性関レイヤー
//======================================
#ifndef __ACTIVATION_BASE_H__
#define __ACTIVATION_BASE_H__

#include<Layer/NeuralNetwork/INNSingle2SingleLayer.h>

#include<vector>

#include"Activation_DATA.hpp"

#include"Activation_LayerData_Base.h"
#include"../_LayerBase/CLayerBase.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	typedef float NEURON_TYPE;	/**< ニューロンに使用するデータ型. float or double */

	class Activation_Base : public CNNSingle2SingleLayerBase
	{
	protected:
	public:
		/** コンストラクタ */
		Activation_Base(Gravisbell::GUID guid, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct);

		/** デストラクタ */
		virtual ~Activation_Base();

		//===========================
		// レイヤー共通
		//===========================
	public:




		//===========================
		// 固有関数
		//===========================
	public:


	};

}	// NeuralNetwork
}	// Layer
}	// Gravisbell


#endif
