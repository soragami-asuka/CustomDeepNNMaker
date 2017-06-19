/*--------------------------------------------
 * FileName  : FullyConnect_DATA.hpp
 * LayerName : �S�������C���[
 * guid      : 14CC33F4-8CD3-4686-9C48-EF452BA5D202
 * 
 * Text      : �S�������C���[.
--------------------------------------------*/
#ifndef __GRAVISBELL_NEURALNETWORK_LAYER_DATA_FullyConnect_H__
#define __GRAVISBELL_NEURALNETWORK_LAYER_DATA_FullyConnect_H__

#include<guiddef.h>

#include<Common/ErrorCode.h>
#include<SettingData/Standard/IData.h>

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {
namespace FullyConnect {

	/** Layer structure */
	struct LayerStructure
	{
		/** Name : �j���[������
		  * ID   : NeuronCount
		  * Text : ���C���[���̃j���[������.
		  *       : �o�̓o�b�t�@���ɒ�������.
		  */
		S32 NeuronCount;

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

} // FullyConnect
} // NeuralNetwork
} // Layer
} // Gravisbell


#endif // __CUSTOM_DEEP_NN_LAYER_DATA_FullyConnect_H__
