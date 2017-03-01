//=======================================
// �P��o�̓��C���[
//=======================================
#ifndef __I_SINGLE_OUTPUT_LAYER_H__
#define __I_SINGLE_OUTPUT_LAYER_H__

#include"LayerErrorCode.h"
#include"ILayerBase.h"
#include"IODataStruct.h"
#include"IOutputLayer.h"

namespace CustomDeepNNLibrary
{
	/** ���C���[�x�[�X */
	class ISingleOutputLayer : public virtual IOutputLayer
	{
	public:
		/** �R���X�g���N�^ */
		ISingleOutputLayer(){}
		/** �f�X�g���N�^ */
		virtual ~ISingleOutputLayer(){}

	public:
		/** �o�̓f�[�^�\�����擾���� */
		virtual IODataStruct GetOutputDataStruct()const = 0;

		/** �o�̓o�b�t�@�����擾����. byte���ł͖����f�[�^�̐��Ȃ̂Œ��� */
		virtual unsigned int GetOutputBufferCount()const = 0;

		/** �o�̓f�[�^�o�b�t�@���擾����.
			�z��̗v�f����[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]
			@return �o�̓f�[�^�z��̐擪�|�C���^ */
		virtual CONST_BATCH_BUFFER_POINTER GetOutputBuffer()const = 0;
		/** �o�̓f�[�^�o�b�t�@���擾����.
			@param o_lpOutputBuffer	�o�̓f�[�^�i�[��z��. [GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v
			@return ���������ꍇ0 */
		virtual ELayerErrorCode GetOutputBuffer(BATCH_BUFFER_POINTER o_lpOutputBuffer)const = 0;
	};
}

#endif