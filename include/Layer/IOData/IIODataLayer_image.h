//=======================================
// ���o�͐M���f�[�^���C���[
//=======================================
#ifndef __GRAVISBELL_I_IO_DATA_LAYER_H__
#define __GRAVISBELL_I_IO_DATA_LAYER_H__

#include"../../Common/Common.h"
#include"../../Common/ErrorCode.h"

#include"IIODataLayer_base.h"


namespace Gravisbell {
namespace Layer {
namespace IOData {

	/** ���o�̓f�[�^���C���[ */
	class IIODataLayer_image : public IIODataLayer_base
	{
	public:
		/** �R���X�g���N�^ */
		IIODataLayer_image(){}
		/** �f�X�g���N�^ */
		virtual ~IIODataLayer_image(){}

	public:
		/** �f�[�^��ǉ�����.
			@param	lpData	�f�[�^��g�̔z��. [GetBufferSize()�̖߂�l]�̗v�f�����K�v. */
		virtual ErrorCode SetData(U32 i_dataNum, const BYTE lpData[], U32 i_lineLength) = 0;
		/** �f�[�^��ԍ��w��Ŏ擾����.
			@param num		�擾����ԍ�
			@param o_LpData �f�[�^�̊i�[��z��. [GetBufferSize()�̖߂�l]�̗v�f�����K�v. */
		virtual ErrorCode GetDataByNum(U32 num, BYTE o_lpData[])const = 0;


		/** �o�b�`�����f�[�^�ԍ����X�g��ݒ肷��.
			�ݒ肳�ꂽ�l������GetDInputBuffer(),GetOutputBuffer()�̖߂�l�����肷��.
			@param i_lpBatchDataNoList	�ݒ肷��f�[�^�ԍ����X�g. [GetBatchSize()�̖߂�l]�̗v�f�����K�v */
		virtual ErrorCode SetBatchDataNoList(const U32 i_lpBatchDataNoList[]) = 0;

	};

}	// IOData
}	// Layer
}	// Gravisbell

#endif