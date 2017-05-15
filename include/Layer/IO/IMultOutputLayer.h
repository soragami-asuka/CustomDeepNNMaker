//=======================================
// �����o�̓��C���[
//=======================================
#ifndef __GRAVISBELL_I_MULT_OUTPUT_LAYER_H__
#define __GRAVISBELL_I_MULT_OUTPUT_LAYER_H__

#include"../../Common/ErrorCode.h"
#include"../../Common/IODataStruct.h"

#include"../ILayerBase.h"
#include"IOutputLayer.h"

namespace Gravisbell {
namespace Layer {
namespace IO {

	/** ���C���[�x�[�X */
	class IMultOutputLayer : public virtual IOutputLayer
	{
	public:
		/** �R���X�g���N�^ */
		IMultOutputLayer(){}
		/** �f�X�g���N�^ */
		virtual ~IMultOutputLayer(){}

	public:
		/** �o�̓f�[�^�̐����擾���� */
		virtual U32 GetOutputDataCount()const = 0;

		/** �o�̓f�[�^�\�����擾����.
			@return	�o�̓f�[�^�\�� */
		virtual IODataStruct GetOutputDataStruct(U32 i_dataNum)const = 0;

		/** �o�̓o�b�t�@�����擾����. byte���ł͖����f�[�^�̐��Ȃ̂Œ��� */
		virtual U32 GetOutputBufferCount(U32 i_dataNum)const = 0;

		/** �o�̓f�[�^�o�b�t�@���擾����.
			�z��̗v�f����[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]
			@return �o�̓f�[�^�z��̐擪�|�C���^ */
		virtual CONST_BATCH_BUFFER_POINTER GetOutputBuffer(U32 i_dataNum)const = 0;
		/** �o�̓f�[�^�o�b�t�@���擾����.
			@param o_lpOutputBuffer	�o�̓f�[�^�i�[��z��. [GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v
			@return ���������ꍇ0 */
		virtual ErrorCode GetOutputBuffer(U32 i_layerNum, BATCH_BUFFER_POINTER o_lpOutputBuffer)const = 0;
	};

}	// IO
}	// Layer
}	// GravisBell

#endif