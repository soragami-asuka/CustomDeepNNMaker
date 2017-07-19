/*--------------------------------------------
 * FileName  : BatchNormalization_DATA.hpp
 * LayerName : �o�b�`���K��
 * guid      : ACD11A5A-BFB5-4951-8382-1DE89DFA96A8
 * 
 * Text      : �o�b�`�P�ʂŐ��K�����s��
--------------------------------------------*/
#ifndef __GRAVISBELL_NEURALNETWORK_LAYER_DATA_BatchNormalization_H__
#define __GRAVISBELL_NEURALNETWORK_LAYER_DATA_BatchNormalization_H__

#include<guiddef.h>

#include<Common/ErrorCode.h>
#include<SettingData/Standard/IData.h>

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {
namespace BatchNormalization {

	/** Layer structure */
	struct LayerStructure
	{
		/** Name : ���艻�W��
		  * ID   : epsilon
		  * Text : ���U�̒l������������ꍇ�Ɋ���Z�����肳���邽�߂̒l
		  */
		F32 epsilon;

		/** Name : ���̓`�����l����
		  * ID   : InputChannelCount
		  * Text : ���̓`�����l����
		  */
		S32 InputChannelCount;

	};

	/** Runtime Parameter structure */
	struct RuntimeParameterStructure
	{
		/** Name : �ŏ����ϒl�X�V�W��
		  * ID   : AverageUpdateCoeffMin
		  * Text : ���ϒl���X�V����ۂ̌W���̍ŏ��l.0=Epoch�̑S�f�[�^�̕��ϒl���g�p����.1=���߂̃f�[�^�̕��ϒl���g�p����.
		  */
		F32 AverageUpdateCoeffMin;

	};

} // BatchNormalization
} // NeuralNetwork
} // Layer
} // Gravisbell


#endif // __CUSTOM_DEEP_NN_LAYER_DATA_BatchNormalization_H__
