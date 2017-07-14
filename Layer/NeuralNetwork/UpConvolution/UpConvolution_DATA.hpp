/*--------------------------------------------
 * FileName  : UpConvolution_DATA.hpp
 * LayerName : 拡張畳みこみニューラルネットワーク
 * guid      : B87B2A75-7EA3-4960-9E9C-EAF43AB073B0
 * 
 * Text      : フィルタ移動量を[Stride/UpScale]に拡張した畳み込みニューラルネットワーク.Stride=1,UpScale=2とした場合、特徴マップのサイズは2倍になる
--------------------------------------------*/
#ifndef __GRAVISBELL_NEURALNETWORK_LAYER_DATA_UpConvolution_H__
#define __GRAVISBELL_NEURALNETWORK_LAYER_DATA_UpConvolution_H__

#include<guiddef.h>

#include<Common/ErrorCode.h>
#include<SettingData/Standard/IData.h>

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {
namespace UpConvolution {

	/** Layer structure */
	struct LayerStructure
	{
		/** Name : フィルタサイズ
		  * ID   : FilterSize
		  * Text : 畳みこみを行う入力信号数
		  */
		Vector3D<S32> FilterSize;

		/** Name : 出力チャンネル数
		  * ID   : Output_Channel
		  * Text : 出力されるチャンネルの数
		  */
		S32 Output_Channel;

		/** Name : フィルタ移動量
		  * ID   : Stride
		  * Text : 畳みこみごとに移動するフィルタの移動量
		  */
		Vector3D<S32> Stride;

		/** Name : 拡張幅
		  * ID   : UpScale
		  */
		Vector3D<S32> UpScale;

		/** Name : パディングサイズ
		  * ID   : Padding
		  */
		Vector3D<S32> Padding;

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

		}PaddingType;

	};

} // UpConvolution
} // NeuralNetwork
} // Layer
} // Gravisbell


#endif // __CUSTOM_DEEP_NN_LAYER_DATA_UpConvolution_H__
