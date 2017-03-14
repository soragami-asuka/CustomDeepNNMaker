/*--------------------------------------------
 * FileName  : Feedforward_DATA.hpp
 * LayerName : 全結合ニューラルネットワークレイヤー
 * guid      : BEBA34EC-C30C-4565-9386-56088981D2D7
 * 
 * Text      : 全結合ニューラルネットワークレイヤー.
 *           : 結合層と活性化層を一体化.
 *           : 学習時に[学習係数][ドロップアウト率]を設定できる.
--------------------------------------------*/
#ifndef __GRAVISBELL_NEURALNETWORK_LAYER_DATA_Feedforward_H__
#define __GRAVISBELL_NEURALNETWORK_LAYER_DATA_Feedforward_H__

#include<guiddef.h>

#include<Common/ErrorCode.h>
#include<SettingData/Standard/IData.h>
#include<Layer/NeuralNetwork/INNLayer.h>

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {
namespace Feedforward {

	/** Layer structure */
	struct LayerStructure
	{
		/** Name : ニューロン数
		  * ID   : NeuronCount
		  * Text : レイヤー内のニューロン数.
		  *       : 出力バッファ数に直結する.
		  */
		int NeuronCount;

		/** Name : 活性化関数種別
		  * ID   : ActivationType
		  * Text : 使用する活性化関数の種類を定義する
		  */
		enum : S32{
			/** Name : シグモイド関数
			  * ID   : sigmoid
			  * Text : y = 1 / (1 + e^(-x));
			  */
			ActivationType_sigmoid,

			/** Name : ReLU（ランプ関数）
			  * ID   : ReLU
			  * Text : y = max(0, x);
			  */
			ActivationType_ReLU,

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

	/** Learning data structure */
	struct LearnDataStructure
	{
		/** Name : 学習係数
		  * ID   : LearnCoeff
		  */
		float LearnCoeff;

		/** Name : ドロップアウト率
		  * ID   : DropOut
		  * Text : 前レイヤーを無視する割合.
		  *       : 1.0で前レイヤーの全出力を無視する
		  */
		float DropOut;

	};

} // Feedforward
} // NeuralNetwork
} // Layer
} // Gravisbell


#endif // __CUSTOM_DEEP_NN_LAYER_DATA_Feedforward_H__
