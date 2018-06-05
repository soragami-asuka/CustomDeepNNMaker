/*--------------------------------------------
 * FileName  : Convolution_DATA.hpp
 * LayerName : 畳みこみニューラルネットワーク
 * guid      : F6662E0E-1CA4-4D59-ACCA-CAC29A16C0AA
 * 
 * Text      : 畳みこみニューラルネットワーク.
--------------------------------------------*/
#ifndef __GRAVISBELL_NEURALNETWORK_LAYER_DATA_Convolution_H__
#define __GRAVISBELL_NEURALNETWORK_LAYER_DATA_Convolution_H__

#include<guiddef.h>

#include<Common/ErrorCode.h>
#include<SettingData/Standard/IData.h>

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {
namespace Convolution {

	/** Layer structure */
	struct LayerStructure
	{
		/** Name : フィルタサイズ
		  * ID   : FilterSize
		  * Text : 畳みこみを行う入力信号数
		  */
		Vector3D<S32> FilterSize;

		/** Name : 入力チャンネル数
		  * ID   : Input_Channel
		  * Text : 入力チャンネル数
		  */
		S32 Input_Channel;

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

		/** Name : 入力拡張量
		  * ID   : Dilation
		  * Text : 入力信号のスキップ幅
		  */
		Vector3D<S32> Dilation;

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

		/** Name : 初期化関数
		  * ID   : Initializer
		  * Text : 初期化関数の種類
		  */
		const wchar_t* Initializer;

		/** Name : 重みデータの種別
		  * ID   : WeightData
		  * Text : 重みデータの種別
		  */
		const wchar_t* WeightData;

	};

	/** Runtime Parameter structure */
	struct RuntimeParameterStructure
	{
		/** Name : 出力の分散を用いて重みを更新するフラグ
		  * ID   : UpdateWeigthWithOutputVariance
		  * Text : 出力の分散を用いて重みを更新するフラグ.trueにした場合Calculate時に出力の分散が1になるまで重みを更新する.
		  */
		bool UpdateWeigthWithOutputVariance;

	};

} // Convolution
} // NeuralNetwork
} // Layer
} // Gravisbell


#endif // __CUSTOM_DEEP_NN_LAYER_DATA_Convolution_H__
