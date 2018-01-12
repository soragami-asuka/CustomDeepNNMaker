/*--------------------------------------------
 * FileName  : MergeAverage_DATA.hpp
 * LayerName : レイヤーのマージ(平均)
 * guid      : 4E993B4B-9F7A-4CEF-A4C4-37B916BFD9B2
 * 
 * Text      : 入力信号のCHを平均して出力する.各入力のX,Y,Zはすべて同一である必要がある.chが不足する部分は0を入力されたものとして扱う
--------------------------------------------*/
#ifndef __GRAVISBELL_NEURALNETWORK_LAYER_DATA_MergeAverage_H__
#define __GRAVISBELL_NEURALNETWORK_LAYER_DATA_MergeAverage_H__

#include<guiddef.h>

#include<Common/ErrorCode.h>
#include<SettingData/Standard/IData.h>

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {
namespace MergeAverage {

	/** Layer structure */
	struct LayerStructure
	{
		/** Name : マージ種別
		  * ID   : MergeType
		  * Text : マージする際にCH数をどのように決定するか
		  */
		enum : S32{
			/** Name : 最大
			  * ID   : max
			  * Text : 入力レイヤーの最大数に併せる
			  */
			MergeType_max,

			/** Name : 最小
			  * ID   : min
			  * Text : 入力レイヤーの最小数に併せる
			  */
			MergeType_min,

			/** Name : 先頭レイヤー
			  * ID   : layer0
			  * Text : 先頭レイヤーの数に併せる
			  */
			MergeType_layer0,

		}MergeType;

	};

} // MergeAverage
} // NeuralNetwork
} // Layer
} // Gravisbell


#endif // __CUSTOM_DEEP_NN_LAYER_DATA_MergeAverage_H__
