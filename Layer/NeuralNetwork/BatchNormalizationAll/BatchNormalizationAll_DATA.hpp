/*--------------------------------------------
 * FileName  : BatchNormalizationAll_DATA.hpp
 * LayerName : �o�b�`���K��(�`�����l����ʂȂ�)
 * guid      : 8AECB925-8DCF-4876-BA6A-6ADBE280D285
 * 
 * Text      : �`�����l���̋�ʂȂ��o�b�`�P�ʂŐ��K�����s��.(�� 1ch�̃f�[�^�Ƃ��Ĉ���)
--------------------------------------------*/
#ifndef __GRAVISBELL_NEURALNETWORK_LAYER_DATA_BatchNormalizationAll_H__
#define __GRAVISBELL_NEURALNETWORK_LAYER_DATA_BatchNormalizationAll_H__

#include<guiddef.h>

#include<Common/ErrorCode.h>
#include<SettingData/Standard/IData.h>

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {
namespace BatchNormalizationAll {

	/** Layer structure */
	struct LayerStructure
	{
		/** Name : ���艻�W��
		  * ID   : epsilon
		  * Text : ���U�̒l������������ꍇ�Ɋ���Z�����肳���邽�߂̒l
		  */
		F32 epsilon;

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

} // BatchNormalizationAll
} // NeuralNetwork
} // Layer
} // Gravisbell


#endif // __CUSTOM_DEEP_NN_LAYER_DATA_BatchNormalizationAll_H__
