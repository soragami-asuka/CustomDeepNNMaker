/*--------------------------------------------
 * FileName  : Value2SignalArray_DATA.hpp
 * LayerName : 信号の配列から値へ変換
 * guid      : 6F6C75B8-9C41-43EA-8F80-98C6F1CF4A2D
 * 
 * Text      : 信号の配列から値へ変換する.
 *           : 最大値を取るCH番号を値に変換する.
 *           : 入力CH数＝分解能の整数倍である必要がある.
--------------------------------------------*/
#ifndef __GRAVISBELL_NEURALNETWORK_LAYER_DATA_Value2SignalArray_H__
#define __GRAVISBELL_NEURALNETWORK_LAYER_DATA_Value2SignalArray_H__

#include<guiddef.h>

#include<Common/ErrorCode.h>
#include<SettingData/Standard/IData.h>

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {
namespace Value2SignalArray {

	/** Layer structure */
	struct LayerStructure
	{
		/** Name : 入力最小値
		  * ID   : inputMinValue
		  */
		F32 inputMinValue;

		/** Name : 入力最大値
		  * ID   : inputMaxValue
		  */
		F32 inputMaxValue;

		/** Name : 分解能
		  * ID   : resolution
		  */
		S32 resolution;

	};

} // Value2SignalArray
} // NeuralNetwork
} // Layer
} // Gravisbell


#endif // __CUSTOM_DEEP_NN_LAYER_DATA_Value2SignalArray_H__
