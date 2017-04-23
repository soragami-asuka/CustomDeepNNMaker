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
#include<Layer/NeuralNetwork/INNLayer.h>

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {
namespace Convolution {

	/** Layer structure */
	struct LayerStructure
	{
		/** Name : 出力チャンネル数
		  * ID   : Output_Channel
		  * Text : 出力されるチャンネルの数
		  */
		S32 Output_Channel;

		/** Name : ドロップアウト率
		  * ID   : DropOut
		  * Text : 前レイヤーを無視する割合.
		  *       : 1.0で前レイヤーの全出力を無視する
		  */
		F32 DropOut;

		/** Name : フィルタサイズ
		  * ID   : FilterSize
		  * Text : 畳みこみを行う入力信号数
		  */
		Vector3D<S32> FilterSize;

		/** Name : フィルタ移動量
		  * ID   : Move
		  * Text : 1ニューロンごとに移動する入力信号の移動量
		  */
		Vector3D<F32> Move;

		/** Name : 畳みこみ移動量
		  * ID   : Stride
		  * Text : 畳みこみごとに移動する入力信号の移動量
		  */
		Vector3D<F32> Stride;

		/** Name : パディングサイズ(-方向)
		  * ID   : PaddingM
		  */
		Vector3D<S32> PaddingM;

		/** Name : パディングサイズ(+方向)
		  * ID   : PaddingP
		  */
		Vector3D<S32> PaddingP;

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

			/** Name : 境界値
			  * ID   : border
			  * Text : 不足分と隣接する値を参照する
			  */
			PaddingType_border,

			/** Name : 反転
			  * ID   : mirror
			  * Text : 不足分と隣接する値から逆方向に参照する
			  */
			PaddingType_mirror,

			/** Name : クランプ
			  * ID   : clamp
			  * Text : 不足分の反対側の境目から順方向に参照する
			  */
			PaddingType_clamp,

		}PaddingType;

	};

	/** Learning data structure */
	struct LearnDataStructure
	{
		/** Name : 学習係数
		  * ID   : LearnCoeff
		  */
		F32 LearnCoeff;

	};

} // Convolution
} // NeuralNetwork
} // Layer
} // Gravisbell


#endif // __CUSTOM_DEEP_NN_LAYER_DATA_Convolution_H__
