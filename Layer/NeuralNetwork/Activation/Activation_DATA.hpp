/*--------------------------------------------
 * FileName  : Activation_DATA.hpp
 * LayerName : �������֐�
 * guid      : 99904134-83B7-4502-A0CA-728A2C9D80C7
 * 
 * Text      : �������֐�
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
		/** Name : �������֐����
		  * ID   : ActivationType
		  * Text : �g�p���銈�����֐��̎�ނ��`����
		  */
		enum : S32{
			/** Name : ���j�A�֐�
			  * ID   : lenear
			  * Text : y = x;
			  */
			ActivationType_lenear,

			/** Name : �V�O���C�h�֐�
			  * ID   : sigmoid
			  * Text : y = 1 / (1 + e^(-x));
			  */
			ActivationType_sigmoid,

			/** Name : �V�O���C�h�֐�(�o�̓��C���[�p)
			  * ID   : sigmoid_crossEntropy
			  * Text : y = 1 / (1 + e^(-x));
			  */
			ActivationType_sigmoid_crossEntropy,

			/** Name : ReLU�i�����v�֐��j
			  * ID   : ReLU
			  * Text : y = max(0, x);
			  */
			ActivationType_ReLU,

			/** Name : SoftMax�֐�
			  * ID   : softmax_ALL
			  * Text : �S�̂ɂ����鎩�g�̊�����Ԃ��֐�.
			  */
			ActivationType_softmax_ALL,

			/** Name : SoftMax�֐�(�o�̓��C���[�p)
			  * ID   : softmax_ALL_crossEntropy
			  * Text : �S�̂ɂ����鎩�g�̊�����Ԃ��֐�.
			  */
			ActivationType_softmax_ALL_crossEntropy,

			/** Name : SoftMax�֐�(CH���̂�)
			  * ID   : softmax_CH
			  * Text : �����X,Y,Z�ɂ�����eCH�̎��g�̊�����Ԃ��֐�.
			  */
			ActivationType_softmax_CH,

			/** Name : SoftMax�֐�(CH���̂�)(�o�̓��C���[�p)
			  * ID   : softmax_CH_crossEntropy
			  * Text : �����X,Y,Z�ɂ�����eCH�̎��g�̊�����Ԃ��֐�.
			  */
			ActivationType_softmax_CH_crossEntropy,

		}ActivationType;

	};

} // Activation
} // NeuralNetwork
} // Layer
} // Gravisbell


#endif // __CUSTOM_DEEP_NN_LAYER_DATA_Activation_H__
