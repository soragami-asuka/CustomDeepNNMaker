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

	/** Runtime Parameter structure */
	struct RuntimeParameterStructure
	{
		/** Name : ドロップアウトを使用するフラグ
		  * ID   : UseDropOut
		  * Text : ドロップアウトを使用するフラグ.trueの場合確率でドロップアウト.falseの場合係数を掛けた値.
		  */
		bool UseDropOut;

	};

} // Dropout
} // NeuralNetwork
} // Layer
} // Gravisbell


#endif // __CUSTOM_DEEP_NN_LAYER_DATA_Dropout_H__
