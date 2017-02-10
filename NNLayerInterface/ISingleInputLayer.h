//=======================================
// �P����̓��C���[
//=======================================
#ifndef __I_SINGLE_INPUT_LAYER_H__
#define __I_SINGLE_INPUT_LAYER_H__

#include"LayerErrorCode.h"
#include"ILayerBase.h"
#include"IODataStruct.h"
#include"IInputLayer.h"

namespace CustomDeepNNLibrary
{
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
		virtual const float** GetDInputBuffer()const = 0;
		/** �w�K�������擾����.
			@param lpDOutputBuffer	�w�K�������i�[����z��.[GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]�̔z�񂪕K�v */
		virtual ELayerErrorCode GetDInputBuffer(float** o_lpDInputBuffer)const = 0;

	public:
		/** ���̓f�[�^�\�����擾����.
			@return	���̓f�[�^�\�� */
		virtual const IODataStruct GetInputDataStruct()const = 0;
		/** ���̓f�[�^�\�����擾����
			@param	o_inputDataStruct	���̓f�[�^�\���̊i�[��
			@return	���������ꍇ0 */
		virtual ELayerErrorCode GetInputDataStruct(IODataStruct& o_inputDataStruct)const = 0;
	};
}

#endif