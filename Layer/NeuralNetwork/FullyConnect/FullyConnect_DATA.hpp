/*--------------------------------------------
 * FileName  : FullyConnect_DATA.hpp
 * LayerName : 全結合レイヤー
 * guid      : 14CC33F4-8CD3-4686-9C48-EF452BA5D202
 * 
 * Text      : 全結合レイヤー.
--------------------------------------------*/
#ifndef __GRAVISBELL_NEURALNETWORK_LAYER_DATA_FullyConnect_H__
#define __GRAVISBELL_NEURALNETWORK_LAYER_DATA_FullyConnect_H__

#include<guiddef.h>

#include<Common/ErrorCode.h>
#include<SettingData/Standard/IData.h>

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {
namespace FullyConnect {

	/** Layer structure */
	struct LayerStructure
	{
		/** Name : 入力バッファ数
		  * ID   : InputBufferCount
		  * Text : レイヤーに対する入力バッファ数
		  */
		S32 InputBufferCount;

		/** Name : ニューロン数
		  * ID   : NeuronCount
		  * Text : レイヤー内のニューロン数.
		  *       : 出力バッファ数に直結する.
		  */
		S32 NeuronCount;

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

} // FullyConnect
} // NeuralNetwork
} // Layer
} // Gravisbell


#endif // __CUSTOM_DEEP_NN_LAYER_DATA_FullyConnect_H__
