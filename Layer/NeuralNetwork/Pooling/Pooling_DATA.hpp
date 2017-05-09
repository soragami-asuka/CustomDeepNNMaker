/*--------------------------------------------
 * FileName  : Pooling_DATA.hpp
 * LayerName : Pooling
 * guid      : EB80E0D0-9D5A-4ED1-A80D-A1667DE0C890
 * 
 * Text      : Pooling.
--------------------------------------------*/
#ifndef __GRAVISBELL_NEURALNETWORK_LAYER_DATA_Pooling_H__
#define __GRAVISBELL_NEURALNETWORK_LAYER_DATA_Pooling_H__

#include<guiddef.h>

#include<Common/ErrorCode.h>
#include<SettingData/Standard/IData.h>
#include<Layer/NeuralNetwork/INNLayer.h>

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {
namespace Pooling {

	/** Layer structure */
	struct LayerStructure
	{
		/** Name : フィルタサイズ
		  * ID   : FilterSize
		  * Text : Poolingを行う範囲
		  */
		Vector3D<S32> FilterSize;

		/** Name : フィルタ移動量
		  * ID   : Stride
		  * Text : 畳みこみごとに移動するフィルタの移動量
		  */
		Vector3D<S32> Stride;

		/** Name : Pooling種別
		  * ID   : PoolingType
		  * Text : Poolingの方法設定
		  */
		enum : S32{
			/** Name : MAXプーリング
			  * ID   : max
			  * Text : 範囲内の最大値を使用する
			  */
			PoolingType_max,

		}PoolingType;

	};

} // Pooling
} // NeuralNetwork
} // Layer
} // Gravisbell


#endif // __CUSTOM_DEEP_NN_LAYER_DATA_Pooling_H__
