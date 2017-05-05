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
#include<Layer/NeuralNetwork/INNLayer.h>

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {
namespace FullyConnect {

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

	};

	/** Learning data structure */
	struct LearnDataStructure
	{
		/** Name : 学習係数
		  * ID   : LearnCoeff
		  */
		F32 LearnCoeff;

	};

} // FullyConnect
} // NeuralNetwork
} // Layer
} // Gravisbell


#endif // __CUSTOM_DEEP_NN_LAYER_DATA_FullyConnect_H__
