/*--------------------------------------------
 * FileName  : Activation_DATA.hpp
 * LayerName : 活性化関数
 * guid      : 99904134-83B7-4502-A0CA-728A2C9D80C7
 * 
 * Text      : 活性化関数
--------------------------------------------*/
#ifndef __GRAVISBELL_NEURALNETWORK_LAYER_DATA_Activation_H__
#define __GRAVISBELL_NEURALNETWORK_LAYER_DATA_Activation_H__

#include<guiddef.h>

#include<Common/ErrorCode.h>
#include<SettingData/Standard/IData.h>
#include<Layer/NeuralNetwork/INNLayer.h>

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {
namespace Activation {

	/** Layer structure */
	struct LayerStructure
	{
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
			  */
			ActivationType_sigmoid,

			/** Name : シグモイド関数(出力レイヤー用)
			  * ID   : sigmoid_crossEntropy
			  * Text : y = 1 / (1 + e^(-x));
			  */
			ActivationType_sigmoid_crossEntropy,

			/** Name : ReLU（ランプ関数）
			  * ID   : ReLU
			  * Text : y = max(0, x);
			  */
			ActivationType_ReLU,

			/** Name : SoftMax関数
			  * ID   : softmax_ALL
			  * Text : 全体における自身の割合を返す関数.
			  */
			ActivationType_softmax_ALL,

			/** Name : SoftMax関数(出力レイヤー用)
			  * ID   : softmax_ALL_crossEntropy
			  * Text : 全体における自身の割合を返す関数.
			  */
			ActivationType_softmax_ALL_crossEntropy,

			/** Name : SoftMax関数(CH内のみ)
			  * ID   : softmax_CH
			  * Text : 同一のX,Y,Zにおける各CHの自身の割合を返す関数.
			  */
			ActivationType_softmax_CH,

			/** Name : SoftMax関数(CH内のみ)(出力レイヤー用)
			  * ID   : softmax_CH_crossEntropy
			  * Text : 同一のX,Y,Zにおける各CHの自身の割合を返す関数.
			  */
			ActivationType_softmax_CH_crossEntropy,

		}ActivationType;

	};

} // Activation
} // NeuralNetwork
} // Layer
} // Gravisbell


#endif // __CUSTOM_DEEP_NN_LAYER_DATA_Activation_H__
