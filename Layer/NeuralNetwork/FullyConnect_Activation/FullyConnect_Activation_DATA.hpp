/*--------------------------------------------
 * FileName  : FullyConnect_Activation_DATA.hpp
 * LayerName : �S�����j���[�����l�b�g���[�N���C���[(�������֐��t��)
 * guid      : BEBA34EC-C30C-4565-9386-56088981D2D7
 * 
 * Text      : �S�����j���[�����l�b�g���[�N���C���[.
 *           : �����w�Ɗ������w����̉�.
 *           : �w�K����[�w�K�W��][�h���b�v�A�E�g��]��ݒ�ł���.
--------------------------------------------*/
#ifndef __GRAVISBELL_NEURALNETWORK_LAYER_DATA_FullyConnect_Activation_H__
#define __GRAVISBELL_NEURALNETWORK_LAYER_DATA_FullyConnect_Activation_H__

#include<guiddef.h>

#include<Common/ErrorCode.h>
#include<SettingData/Standard/IData.h>
#include<Layer/NeuralNetwork/INNLayer.h>

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {
namespace FullyConnect_Activation {

	/** Layer structure */
	struct LayerStructure
	{
		/** Name : �j���[������
		  * ID   : NeuronCount
		  * Text : ���C���[���̃j���[������.
		  *       : �o�̓o�b�t�@���ɒ�������.
		  */
		int NeuronCount;

		/** Name : �h���b�v�A�E�g��
		  * ID   : DropOut
		  * Text : �O���C���[�𖳎����銄��.
		  *       : 1.0�őO���C���[�̑S�o�͂𖳎�����
		  */
		float DropOut;

		/** Name : �������֐����
		  * ID   : ActivationType
		  * Text : �g�p���銈�����֐��̎�ނ��`����
		  */
		enum : S32{
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
			  * ID   : softmax
			  * Text : �S�̂ɂ����鎩�g�̊�����Ԃ��֐�.
			  */
			ActivationType_softmax,

			/** Name : SoftMax�֐�(�o�̓��C���[�p)
			  * ID   : softmax_crossEntropy
			  * Text : �S�̂ɂ����鎩�g�̊�����Ԃ��֐�.
			  */
			ActivationType_softmax_crossEntropy,

		}ActivationType;

		/** Name : Bool�^�̃T���v��
		  * ID   : BoolSample
		  */
		bool BoolSample;

		/** Name : String�^�̃T���v��
		  * ID   : StringSample
		  */
		const wchar_t* StringSample;

	};

	/** Learning data structure */
	struct LearnDataStructure
	{
		/** Name : �w�K�W��
		  * ID   : LearnCoeff
		  */
		float LearnCoeff;

	};

} // FullyConnect_Activation
} // NeuralNetwork
} // Layer
} // Gravisbell


#endif // __CUSTOM_DEEP_NN_LAYER_DATA_FullyConnect_Activation_H__
