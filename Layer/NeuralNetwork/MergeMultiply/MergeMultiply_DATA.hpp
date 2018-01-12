/*--------------------------------------------
 * FileName  : MergeMultiply_DATA.hpp
 * LayerName : レイヤーのマージ(乗算)
 * guid      : 7BECF785-60D0-42AD-8857-1E2CB2D56721
 * 
 * Text      : 入力信号のCHを乗算して出力する.各入力のX,Y,Zはすべて同一である必要がある
--------------------------------------------*/
#ifndef __GRAVISBELL_NEURALNETWORK_LAYER_DATA_MergeMultiply_H__
#define __GRAVISBELL_NEURALNETWORK_LAYER_DATA_MergeMultiply_H__

#include<guiddef.h>

#include<Common/ErrorCode.h>
#include<SettingData/Standard/IData.h>

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {
namespace MergeMultiply {

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

} // MergeMultiply
} // NeuralNetwork
} // Layer
} // Gravisbell


#endif // __CUSTOM_DEEP_NN_LAYER_DATA_MergeMultiply_H__
