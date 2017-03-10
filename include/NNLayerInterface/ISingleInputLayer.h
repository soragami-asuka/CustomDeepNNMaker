//=======================================
// �P����̓��C���[
//=======================================
#ifndef __GRAVISBELL_I_SINGLE_INPUT_LAYER_H__
#define __GRAVISBELL_I_SINGLE_INPUT_LAYER_H__

#include"Common/ErrorCode.h"
#include"Common/IODataStruct.h"

#include"ILayerBase.h"
#include"IInputLayer.h"

namespace Gravisbell {
namespace NeuralNetwork {

	/** ���C���[�x�[�X */
	class ISingleInputLayer : public virtual IInputLayer
	{
	public:
		/** �R���X�g���N�^ */
		ISingleInputLayer(){}
		/** �f�X�g���N�^ */
		virtual ~ISingleInputLayer(){}

	public:
		/** ���̓o�b�t�@�����擾����. byte���ł͖����f�[�^�̐��Ȃ̂Œ��� */
		virtual unsigned int GetInputBufferCount()const = 0;

		/** �w�K�������擾����.
			�z��̗v�f����[GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]
			@return	�덷�����z��̐擪�|�C���^ */
		virtual CONST_BATCH_BUFFER_POINTER GetDInputBuffer()const = 0;
		/** �w�K�������擾����.
			@param lpDInputBuffer	�w�K�������i�[����z��.[GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]�̔z�񂪕K�v */
		virtual ErrorCode GetDInputBuffer(BATCH_BUFFER_POINTER o_lpDInputBuffer)const = 0;

	public:
		/** ���̓f�[�^�\�����擾����.
			@return	���̓f�[�^�\�� */
		virtual IODataStruct GetInputDataStruct()const = 0;
	};

}	// NeuralNetwork
}	// Gravisbell

#endif