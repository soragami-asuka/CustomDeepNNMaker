/*--------------------------------------------
 * FileName  : FullyConnect_Activation_DATA.hpp
 * LayerName : 全結合ニューラルネットワークレイヤー(活性化関数付き)
 * guid      : BEBA34EC-C30C-4565-9386-56088981D2D7
 * 
 * Text      : 全結合ニューラルネットワークレイヤー.
 *           : 結合層と活性化層を一体化.
 *           : 学習時に[学習係数][ドロップアウト率]を設定できる.
--------------------------------------------*/
#ifndef __GRAVISBELL_NEURALNETWORK_LAYER_DATA_FullyConnect_Activation_H__
#define __GRAVISBELL_NEURALNETWORK_LAYER_DATA_FullyConnect_Activation_H__

#include<guiddef.h>

#include<Common/ErrorCode.h>
#include<SettingData/Standard/IData.h>

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {
namespace FullyConnect_Activation {

	/** Layer structure */
	struct LayerStructure
	{
		/** Name : ニューロン数
		  * ID   : NeuronCount
		  * Text : レイヤー内のニューロン数.
		  *       : 出力バッファ数に直結する.
		  */
		S32 NeuronCount;

		/** Name : ドロップアウト率
		  * ID   : DropOut
		  * Text : 前レイヤーを無視する割合.
		  *       : 1.0で前レイヤーの全出力を無視する
		  */
		F32 DropOut;

		/** Name : 活性化関数種別
		  * ID   : ActivationType
		  * Text : 使用する活性化関数の種類を定義する
		  */
		enum : S32{
			/** Name : リニア関数
			  * ID   : lenear
			  * Text : y = x;
			  */
			ActivationType_lenear,

			/** Name : シグモイド関数
			  * ID   : sigmoid
			  * Text : y = 1 / (1 + e^(-x));
			  *      : 範囲 0 < y < 1
			  *      : (x=0, y=0.5)を通る
			  */
			ActivationType_sigmoid,

			/** Name : シグモイド関数(出力レイヤー用)
			  * ID   : sigmoid_crossEntropy
			  * Text : y = 1 / (1 + e^(-x));
			  *      : 範囲 0 < y < 1
			  *      : (x=0, y=0.5)を通る
			  */
			ActivationType_sigmoid_crossEntropy,

			/** Name : ReLU（ランプ関数）
			  * ID   : ReLU
			  * Text : y = max(0, x);
			  *      : 範囲 0 <= y
			  *      : (x=0, y=0)を通る
			  */
			ActivationType_ReLU,

			/** Name : SoftMax関数
			  * ID   : softmax
			  * Text : 全体における自身の割合を返す関数.
			  *      : y = e^x / Σe^x;
			  */
			ActivationType_softmax,

			/** Name : SoftMax関数(出力レイヤー用)
			  * ID   : softmax_crossEntropy
			  * Text : 全体における自身の割合を返す関数.
			  *      : y = e^x / Σe^x;
			  */
			ActivationType_softmax_crossEntropy,

		}ActivationType;

		/** Name : Bool型のサンプル
		  * ID   : BoolSample
		  */
		bool BoolSample;

		/** Name : String型のサンプル
		  * ID   : StringSample
		  */
		const wchar_t* StringSample;

	};

} // FullyConnect_Activation
} // NeuralNetwork
} // Layer
} // Gravisbell


#endif // __CUSTOM_DEEP_NN_LAYER_DATA_FullyConnect_Activation_H__
