/*--------------------------------------------
 * FileName  : MergeMax_DATA.hpp
 * LayerName : レイヤーのマージ(最大値)
 * guid      : 3F015946-7E88-4DB0-91BD-F4013F2190D4
 * 
 * Text      : 入力信号のCHの最大値を出力する.各入力のX,Y,Zはすべて同一である必要がある
--------------------------------------------*/
#ifndef __GRAVISBELL_NEURALNETWORK_LAYER_DATA_MergeMax_H__
#define __GRAVISBELL_NEURALNETWORK_LAYER_DATA_MergeMax_H__

#include<guiddef.h>

#include<Common/ErrorCode.h>
#include<SettingData/Standard/IData.h>

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {
namespace MergeMax {

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

} // MergeMax
} // NeuralNetwork
} // Layer
} // Gravisbell


#endif // __CUSTOM_DEEP_NN_LAYER_DATA_MergeMax_H__
