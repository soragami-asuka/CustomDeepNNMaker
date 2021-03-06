/*--------------------------------------------
 * FileName  : FeedforwardNeuralNetwork_DATA.hpp
 * LayerName : 順伝播型ニューラルネットワーク-複数レイヤー結合
 * guid      : 1C38E21F-6F01-41B2-B40E-7F67267A3692
 * 
 * Text      : 順伝播型ニューラルネットワーク.
 *           : 複数レイヤーを一体化し管理するためのクラス.
--------------------------------------------*/
#ifndef __GRAVISBELL_NEURALNETWORK_LAYER_DATA_FeedforwardNeuralNetwork_H__
#define __GRAVISBELL_NEURALNETWORK_LAYER_DATA_FeedforwardNeuralNetwork_H__

#include<guiddef.h>

#include<Common/ErrorCode.h>
#include<SettingData/Standard/IData.h>

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {
namespace FeedforwardNeuralNetwork {

	/** Layer structure */
	struct LayerStructure
	{
		/** Name : 入力レイヤー数
		  * ID   : inputLayerCount
		  * Text : 入力信号として取り扱うことのできるレイヤー数
		  */
		S32 inputLayerCount;

	};

} // FeedforwardNeuralNetwork
} // NeuralNetwork
} // Layer
} // Gravisbell


#endif // __CUSTOM_DEEP_NN_LAYER_DATA_FeedforwardNeuralNetwork_H__
