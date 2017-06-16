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

	/** Learning data structure */
	struct LearnDataStructure
	{
		/** Name : 最適化ルーチン
		  * ID   : Optimizer
		  * Text : 重み誤差を反映させるためのアルゴリズム.
		  */
		enum : S32{
			/** Name : SGD
			  * ID   : SGD
			  */
			Optimizer_SGD,

			/** Name : Momentum
			  * ID   : Momentum
			  * Text : 慣性付与
			  */
			Optimizer_Momentum,

			/** Name : AdaDelta
			  * ID   : AdaDelta
			  */
			Optimizer_AdaDelta,

			/** Name : Adam
			  * ID   : Adam
			  */
			Optimizer_Adam,

		}Optimizer;

		/** Name : 学習係数
		  * ID   : LearnCoeff
		  * Text : SGD,Momentumで使用.
		  */
		F32 LearnCoeff;

		/** Name : Momentum-α
		  * ID   : Momentum_alpha
		  * Text : Momentumで使用.t-1の値を反映する割合
		  */
		F32 Momentum_alpha;

		/** Name : AdaDelta-β
		  * ID   : AdaDelta_beta
		  * Text : AdaDeltaで使用.減衰率.高いほうが減衰しづらい.
		  */
		F32 AdaDelta_beta;

		/** Name : AdaDelta-ε
		  * ID   : AdaDelta_epsilon
		  * Text : AdaDeltaで使用.補助数.高いほど初期値が大きくなる.
		  */
		F32 AdaDelta_epsilon;

		/** Name : Adam-α
		  * ID   : Adam_alpha
		  * Text : Adamで使用.加算率.高いほうが更新されやすい.
		  */
		F32 Adam_alpha;

		/** Name : Adam-β
		  * ID   : Adam_beta1
		  * Text : Adamで使用.減衰率1.高いほうが減衰しづらい.
		  */
		F32 Adam_beta1;

		/** Name : Adam-β
		  * ID   : Adam_beta2
		  * Text : Adamで使用.減衰率2.高いほうが減衰しづらい.
		  */
		F32 Adam_beta2;

		/** Name : Adam-ε
		  * ID   : Adam_epsilon
		  * Text : AdaDeltaで使用.補助数.高いほど初期値が大きくなる.
		  */
		F32 Adam_epsilon;

	};

} // Convolution
} // NeuralNetwork
} // Layer
} // Gravisbell


#endif // __CUSTOM_DEEP_NN_LAYER_DATA_Convolution_H__
