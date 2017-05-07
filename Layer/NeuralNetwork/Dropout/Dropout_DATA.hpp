/*--------------------------------------------
 * FileName  : Dropout_DATA.hpp
 * LayerName : ドロップアウト
 * guid      : 298243E4-2111-474F-A8F4-35BDC8764588
 * 
 * Text      : ドロップアウト.
--------------------------------------------*/
#ifndef __GRAVISBELL_NEURALNETWORK_LAYER_DATA_Dropout_H__
#define __GRAVISBELL_NEURALNETWORK_LAYER_DATA_Dropout_H__

#include<guiddef.h>

#include<Common/ErrorCode.h>
#include<SettingData/Standard/IData.h>
#include<Layer/NeuralNetwork/INNLayer.h>

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {
namespace Dropout {

	/** Layer structure */
	struct LayerStructure
	{
		/** Name : ドロップアウト率
		  * ID   : Rate
		  * Text : 前レイヤーを無視する割合.
		  *       : 1.0で前レイヤーの全出力を無視する
		  */
		F32 Rate;

	};

} // Dropout
} // NeuralNetwork
} // Layer
} // Gravisbell


#endif // __CUSTOM_DEEP_NN_LAYER_DATA_Dropout_H__
