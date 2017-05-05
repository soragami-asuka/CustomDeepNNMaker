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
#include<Layer/NeuralNetwork/INNLayer.h>

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

		/** Name : �h���b�v�A�E�g��
		  * ID   : DropOut
		  * Text : �O���C���[�𖳎����銄��.
		  *       : 1.0�őO���C���[�̑S�o�͂𖳎�����
		  */
		F32 DropOut;

	};

	/** Learning data structure */
	struct LearnDataStructure
	{
		/** Name : �w�K�W��
		  * ID   : LearnCoeff
		  */
		F32 LearnCoeff;

	};

} // FullyConnect
} // NeuralNetwork
} // Layer
} // Gravisbell


#endif // __CUSTOM_DEEP_NN_LAYER_DATA_FullyConnect_H__
