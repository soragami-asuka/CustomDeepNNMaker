/*--------------------------------------------
 * FileName  : ProbabilityArray2Value_DATA.hpp
 * LayerName : 確率の配列から値へ変換
 * guid      : 9E32D735-A29D-4636-A9CE-2C781BA7BE8E
 * 
 * Text      : 確率の配列から値へ変換する.
 *           : 最大値を取るCH番号を値に変換する.
 *           : 入力CH数＝分解能の整数倍である必要がある.
 *           : 学習時の入力に対する教師信号は正解信号を中心とした正規分布の平均値をとる
--------------------------------------------*/
#ifndef __GRAVISBELL_NEURALNETWORK_LAYER_DATA_ProbabilityArray2Value_H__
#define __GRAVISBELL_NEURALNETWORK_LAYER_DATA_ProbabilityArray2Value_H__

#include<guiddef.h>

#include<Common/ErrorCode.h>
#include<SettingData/Standard/IData.h>

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {
namespace ProbabilityArray2Value {

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

		/** Name : 教師信号の分散
		  * ID   : variance
		  */
		F32 variance;

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

} // ProbabilityArray2Value
} // NeuralNetwork
} // Layer
} // Gravisbell


#endif // __CUSTOM_DEEP_NN_LAYER_DATA_ProbabilityArray2Value_H__
