/*--------------------------------------------
 * FileName  : ExponentialNormalization_DATA.hpp
 * LayerName : �o�b�`���K��
 * guid      : 44F733E8-417C-4598-BF05-2CC26E1AB6F1
 * 
 * Text      : �o�b�`�P�ʂŐ��K�����s��
--------------------------------------------*/
#ifndef __GRAVISBELL_NEURALNETWORK_LAYER_DATA_ExponentialNormalization_H__
#define __GRAVISBELL_NEURALNETWORK_LAYER_DATA_ExponentialNormalization_H__

#include<guiddef.h>

#include<Common/ErrorCode.h>
#include<SettingData/Standard/IData.h>

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {
namespace ExponentialNormalization {

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

		/** Name : ���������Ԑ�
		  * ID   : ExponentialTime
		  * Text : ���������Ԑ�
		  */
		S32 ExponentialTime;

		/** Name : ���������Ԑ�
		  * ID   : InitParameterTime
		  * Text : ���������Ԑ�.���������邽�߂Ɏg�p���鎞�Ԑ�.
		  */
		S32 InitParameterTime;

	};

	/** Runtime Parameter structure */
	struct RuntimeParameterStructure
	{
		/** Name : �����W��
		  * ID   : AccelCoeff
		  * Text : �����W��
		  */
		F32 AccelCoeff;

	};

} // ExponentialNormalization
} // NeuralNetwork
} // Layer
} // Gravisbell


#endif // __CUSTOM_DEEP_NN_LAYER_DATA_ExponentialNormalization_H__
