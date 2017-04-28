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
#include<Layer/NeuralNetwork/INNLayer.h>

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

	};

	/** Learning data structure */
	struct LearnDataStructure
	{
		/** Name : �w�K�W��
		  * ID   : LearnCoeff
		  */
		F32 LearnCoeff;

	};

} // BatchNormalization
} // NeuralNetwork
} // Layer
} // Gravisbell


#endif // __CUSTOM_DEEP_NN_LAYER_DATA_BatchNormalization_H__
