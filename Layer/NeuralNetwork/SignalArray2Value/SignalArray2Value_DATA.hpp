/*--------------------------------------------
 * FileName  : SignalArray2Value_DATA.hpp
 * LayerName : 信号の配列から値へ変換
 * guid      : 97C8E5D3-AA0E-43AA-96C2-E7E434F104B8
 * 
 * Text      : 信号の配列から値へ変換する.
 *           : 出力CH数を1に強制変換.
--------------------------------------------*/
#ifndef __GRAVISBELL_NEURALNETWORK_LAYER_DATA_SignalArray2Value_H__
#define __GRAVISBELL_NEURALNETWORK_LAYER_DATA_SignalArray2Value_H__

#include<guiddef.h>

#include<Common/ErrorCode.h>
#include<SettingData/Standard/IData.h>

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {
namespace SignalArray2Value {

	/** Layer structure */
	struct LayerStructure
	{
		/** Name : 出力最小値
		  * ID   : outputMinValue
		  */
		F32 outputMinValue;

		/** Name : 出力最大値
		  * ID   : outputMaxValue
		  */
		F32 outputMaxValue;

	};

} // SignalArray2Value
} // NeuralNetwork
} // Layer
} // Gravisbell


#endif // __CUSTOM_DEEP_NN_LAYER_DATA_SignalArray2Value_H__
