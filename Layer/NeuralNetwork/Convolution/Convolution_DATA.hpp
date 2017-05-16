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

		/** Name : 
		  * ID   : UpScale
		  */
		Vector3D<S32> UpScale;

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
		/** Name : �w�K�W��
		  * ID   : LearnCoeff
		  */
		F32 LearnCoeff;

	};

} // Convolution
} // NeuralNetwork
} // Layer
} // Gravisbell


#endif // __CUSTOM_DEEP_NN_LAYER_DATA_Convolution_H__
