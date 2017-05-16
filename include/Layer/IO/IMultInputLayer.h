//=======================================
// �������̓��C���[
//=======================================
#ifndef __GRAVISBELL_I_MULT_INPUT_LAYER_H__
#define __GRAVISBELL_I_MULT_INPUT_LAYER_H__

#include"../../Common/ErrorCode.h"
#include"../../Common/IODataStruct.h"

#include"../ILayerBase.h"
#include"IInputLayer.h"

namespace Gravisbell {
namespace Layer {
namespace IO {

	/** ���C���[�x�[�X */
	class IMultInputLayer : public virtual IInputLayer
	{
	public:
		/** �R���X�g���N�^ */
		IMultInputLayer(){}
		/** �f�X�g���N�^ */
		virtual ~IMultInputLayer(){}

	public:
		/** ���̓f�[�^�̐����擾���� */
		virtual U32 GetInputDataCount()const = 0;

		/** ���̓f�[�^�\�����擾����.
			@return	���̓f�[�^�\�� */
		virtual IODataStruct GetInputDataStruct(U32 i_dataNum)const = 0;

		/** ���̓o�b�t�@�����擾����. byte���ł͖����f�[�^�̐��Ȃ̂Œ��� */
		virtual U32 GetInputBufferCount(U32 i_dataNum)const = 0;

		/** �w�K�������擾����.
			�z��̗v�f����[GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]
			@return	�덷�����z��̐擪�|�C���^ */
		virtual CONST_BATCH_BUFFER_POINTER GetDInputBuffer(U32 i_dataNum)const = 0;
		/** �w�K�������擾����.
			@param lpDInputBuffer	�w�K�������i�[����z��.[GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]�̔z�񂪕K�v.
			@return ���������ꍇ0 */
		virtual ErrorCode GetDInputBuffer(U32 i_dataNum, BATCH_BUFFER_POINTER o_lpDInputBuffer)const = 0;
	};

}	// IO
}	// Layer
}	// Gravisbell

#endif