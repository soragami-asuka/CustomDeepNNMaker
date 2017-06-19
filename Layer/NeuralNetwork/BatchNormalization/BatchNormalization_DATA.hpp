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

	};

	/** Learning data structure */
	struct LearnDataStructure
	{
		/** Name : �œK�����[�`��
		  * ID   : Optimizer
		  * Text : �d�݌덷�𔽉f�����邽�߂̃A���S���Y��.
		  */
		enum : S32{
			/** Name : SGD
			  * ID   : SGD
			  */
			Optimizer_SGD,

			/** Name : Momentum
			  * ID   : Momentum
			  * Text : �����t�^
			  */
			Optimizer_Momentum,

			/** Name : AdaDelta
			  * ID   : AdaDelta
			  */
			Optimizer_AdaDelta,

			/** Name : Adam
			  * ID   : Adam
			  */
			Optimizer_Adam,

		}Optimizer;

		/** Name : �w�K�W��
		  * ID   : LearnCoeff
		  * Text : SGD,Momentum�Ŏg�p.
		  */
		F32 LearnCoeff;

		/** Name : Momentum-��
		  * ID   : Momentum_alpha
		  * Text : Momentum�Ŏg�p.t-1�̒l�𔽉f���銄��
		  */
		F32 Momentum_alpha;

		/** Name : AdaDelta-��
		  * ID   : AdaDelta_rho
		  * Text : AdaDelta�Ŏg�p.������.�����ق����������Â炢.
		  */
		F32 AdaDelta_rho;

		/** Name : AdaDelta-��
		  * ID   : AdaDelta_epsilon
		  * Text : AdaDelta�Ŏg�p.�⏕��.�����قǏ����l���傫���Ȃ�.
		  */
		F32 AdaDelta_epsilon;

		/** Name : Adam-��
		  * ID   : Adam_alpha
		  * Text : Adam�Ŏg�p.���Z��.�����ق����X�V����₷��.
		  */
		F32 Adam_alpha;

		/** Name : Adam-��
		  * ID   : Adam_beta1
		  * Text : Adam�Ŏg�p.������1.�����ق����������Â炢.
		  */
		F32 Adam_beta1;

		/** Name : Adam-��
		  * ID   : Adam_beta2
		  * Text : Adam�Ŏg�p.������2.�����ق����������Â炢.
		  */
		F32 Adam_beta2;

		/** Name : Adam-��
		  * ID   : Adam_epsilon
		  * Text : AdaDelta�Ŏg�p.�⏕��.�����قǏ����l���傫���Ȃ�.
		  */
		F32 Adam_epsilon;

	};

} // BatchNormalization
} // NeuralNetwork
} // Layer
} // Gravisbell


#endif // __CUSTOM_DEEP_NN_LAYER_DATA_BatchNormalization_H__
