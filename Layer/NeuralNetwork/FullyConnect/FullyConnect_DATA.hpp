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
		/** Name : ���̓o�b�t�@��
		  * ID   : InputBufferCount
		  * Text : ���C���[�ɑ΂�����̓o�b�t�@��
		  */
		S32 InputBufferCount;

		/** Name : �j���[������
		  * ID   : NeuronCount
		  * Text : ���C���[���̃j���[������.
		  *       : �o�̓o�b�t�@���ɒ�������.
		  */
		S32 NeuronCount;

	};

	/** Runtime Parameter structure */
	struct RuntimeParameterStructure
	{
		/** Name : �o�͂̕��U��p���ďd�݂��X�V����t���O
		  * ID   : UpdateWeigthWithOutputVariance
		  * Text : �o�͂̕��U��p���ďd�݂��X�V����t���O.true�ɂ����ꍇCalculate���ɏo�͂̕��U��1�ɂȂ�܂ŏd�݂��X�V����.
		  */
		bool UpdateWeigthWithOutputVariance;

	};

} // FullyConnect
} // NeuralNetwork
} // Layer
} // Gravisbell


#endif // __CUSTOM_DEEP_NN_LAYER_DATA_FullyConnect_H__
