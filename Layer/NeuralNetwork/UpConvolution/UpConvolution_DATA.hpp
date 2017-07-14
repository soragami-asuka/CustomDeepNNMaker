/*--------------------------------------------
 * FileName  : UpConvolution_DATA.hpp
 * LayerName : �g����݂��݃j���[�����l�b�g���[�N
 * guid      : B87B2A75-7EA3-4960-9E9C-EAF43AB073B0
 * 
 * Text      : �t�B���^�ړ��ʂ�[Stride/UpScale]�Ɋg��������ݍ��݃j���[�����l�b�g���[�N.Stride=1,UpScale=2�Ƃ����ꍇ�A�����}�b�v�̃T�C�Y��2�{�ɂȂ�
--------------------------------------------*/
#ifndef __GRAVISBELL_NEURALNETWORK_LAYER_DATA_UpConvolution_H__
#define __GRAVISBELL_NEURALNETWORK_LAYER_DATA_UpConvolution_H__

#include<guiddef.h>

#include<Common/ErrorCode.h>
#include<SettingData/Standard/IData.h>

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {
namespace UpConvolution {

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

		/** Name : �g����
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

} // UpConvolution
} // NeuralNetwork
} // Layer
} // Gravisbell


#endif // __CUSTOM_DEEP_NN_LAYER_DATA_UpConvolution_H__
