/*--------------------------------------------
 * FileName  : SignalArray2Value_DATA.hpp
 * LayerName : 信号の配列から値へ変換
 * guid      : 97C8E5D3-AA0E-43AA-96C2-E7E434F104B8
 * 
 * Text      : 信号の配列から値へ変換する.
 *           : 最大値を取るCH番号を値に変換する.
 *           : 入力CH数＝分解能の整数倍である必要がある.
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

		/** Name : 分解能
		  * ID   : resolution
		  */
		S32 resolution;

		/** Name : 割り当て種別
		  * ID   : allocationType
		  * Text : CH番号→値に変換するための変換方法
		  */
		enum : S32{
			/** Name : 最大値
			  * ID   : max
			  * Text : CH内の最大値を出力する
			  */
			allocationType_max,

			/** Name : 平均
			  * ID   : average
			  * Text : CH番号とCHの値を掛け合わせた値の平均値を出力する(相加平均)
			  */
			allocationType_average,

		}allocationType;

	};

} // SignalArray2Value
} // NeuralNetwork
} // Layer
} // Gravisbell


#endif // __CUSTOM_DEEP_NN_LAYER_DATA_SignalArray2Value_H__
