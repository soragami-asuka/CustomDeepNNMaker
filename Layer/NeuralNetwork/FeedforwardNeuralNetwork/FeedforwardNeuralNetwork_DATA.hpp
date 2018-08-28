/*--------------------------------------------
 * FileName  : FeedforwardNeuralNetwork_DATA.hpp
 * LayerName : ���`�d�^�j���[�����l�b�g���[�N-�������C���[����
 * guid      : 1C38E21F-6F01-41B2-B40E-7F67267A3692
 * 
 * Text      : ���`�d�^�j���[�����l�b�g���[�N.
 *           : �������C���[����̉����Ǘ����邽�߂̃N���X.
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
		/** Name : ���̓��C���[��
		  * ID   : inputLayerCount
		  * Text : ���͐M���Ƃ��Ď�舵�����Ƃ̂ł��郌�C���[��
		  */
		S32 inputLayerCount;

	};

} // FeedforwardNeuralNetwork
} // NeuralNetwork
} // Layer
} // Gravisbell


#endif // __CUSTOM_DEEP_NN_LAYER_DATA_FeedforwardNeuralNetwork_H__
