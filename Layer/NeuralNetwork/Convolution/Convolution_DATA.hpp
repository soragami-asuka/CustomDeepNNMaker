/*--------------------------------------------
 * FileName  : Convolution_DATA.hpp
 * LayerName : ��݂��݃j���[�����l�b�g���[�N
 * guid      : F6662E0E-1CA4-4D59-ACCA-CAC29A16C0AA
 * 
 * Text      : ��݂��݃j���[�����l�b�g���[�N.
--------------------------------------------*/
#ifndef __GRAVISBELL_NEURALNETWORK_LAYER_DATA_Convolution_H__
#define __GRAVISBELL_NEURALNETWORK_LAYER_DATA_Convolution_H__

#include<guiddef.h>

#include<Common/ErrorCode.h>
#include<SettingData/Standard/IData.h>

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {
namespace Convolution {

	/** Layer structure */
	struct LayerStructure
	{
		/** Name : �t�B���^�T�C�Y
		  * ID   : FilterSize
		  * Text : ��݂��݂��s�����͐M����
		  */
		Vector3D<S32> FilterSize;

		/** Name : �o�̓`�����l����
		  * ID   : Output_Channel
		  * Text : �o�͂����`�����l���̐�
		  */
		S32 Output_Channel;

		/** Name : �t�B���^�ړ���
		  * ID   : Stride
		  * Text : ��݂��݂��ƂɈړ�����t�B���^�̈ړ���
		  */
		Vector3D<S32> Stride;

		/** Name : �p�f�B���O�T�C�Y
		  * ID   : Padding
		  */
		Vector3D<S32> Padding;

		/** Name : �p�f�B���O���
		  * ID   : PaddingType
		  * Text : �p�f�B���O���s���ۂ̕��@�ݒ�
		  */
		enum : S32{
			/** Name : �[���p�f�B���O
			  * ID   : zero
			  * Text : �s������0�Ŗ��߂�
			  */
			PaddingType_zero,

		}PaddingType;

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
		  * ID   : AdaDelta_beta
		  * Text : AdaDelta�Ŏg�p.������.�����ق����������Â炢.
		  */
		F32 AdaDelta_beta;

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

} // Convolution
} // NeuralNetwork
} // Layer
} // Gravisbell


#endif // __CUSTOM_DEEP_NN_LAYER_DATA_Convolution_H__
