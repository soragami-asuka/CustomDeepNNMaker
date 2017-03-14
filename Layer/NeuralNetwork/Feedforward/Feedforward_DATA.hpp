/*--------------------------------------------
 * FileName  : Feedforward_DATA.hpp
 * LayerName : �S�����j���[�����l�b�g���[�N���C���[
 * guid      : BEBA34EC-C30C-4565-9386-56088981D2D7
 * 
 * Text      : �S�����j���[�����l�b�g���[�N���C���[.
 *           : �����w�Ɗ������w����̉�.
 *           : �w�K����[�w�K�W��][�h���b�v�A�E�g��]��ݒ�ł���.
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
		/** Name : �j���[������
		  * ID   : NeuronCount
		  * Text : ���C���[���̃j���[������.
		  *       : �o�̓o�b�t�@���ɒ�������.
		  */
		int NeuronCount;

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

			/** Name : ReLU�i�����v�֐��j
			  * ID   : ReLU
			  * Text : y = max(0, x);
			  */
			ActivationType_ReLU,

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

		/** Name : �h���b�v�A�E�g��
		  * ID   : DropOut
		  * Text : �O���C���[�𖳎����銄��.
		  *       : 1.0�őO���C���[�̑S�o�͂𖳎�����
		  */
		float DropOut;

	};

} // Feedforward
} // NeuralNetwork
} // Layer
} // Gravisbell


#endif // __CUSTOM_DEEP_NN_LAYER_DATA_Feedforward_H__
