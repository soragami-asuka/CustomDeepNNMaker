/*--------------------------------------------
 * FileName  : UpSampling_DATA.hpp
 * LayerName : アップサンプリング
 * guid      : 14EEE4A7-1B26-4651-8EBF-B1156D62CE1B
 * 
 * Text      : 値を拡張し、穴埋めする
--------------------------------------------*/
#ifndef __GRAVISBELL_NEURALNETWORK_LAYER_DATA_UpSampling_H__
#define __GRAVISBELL_NEURALNETWORK_LAYER_DATA_UpSampling_H__

#include<guiddef.h>

#include<Common/ErrorCode.h>
#include<SettingData/Standard/IData.h>

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {
namespace UpSampling {

	/** Layer structure */
	struct LayerStructure
	{
		/** Name : 拡張幅
		  * ID   : UpScale
		  */
		Vector3D<S32> UpScale;

		/** Name : パディング種別
		  * ID   : PaddingType
		  * Text : パディングを行う際の方法設定
		  */
		enum : S32{
			/** Name : ゼロパディング
			  * ID   : zero
			  * Text : 不足分を0で埋める
			  */
			PaddingType_zero,

			/** Name : 値
			  * ID   : value
			  * Text : 不足分と隣接する値を参照する
			  */
			PaddingType_value,

		}PaddingType;

	};

} // UpSampling
} // NeuralNetwork
} // Layer
} // Gravisbell


#endif // __CUSTOM_DEEP_NN_LAYER_DATA_UpSampling_H__
