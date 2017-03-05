//=======================================
// �f�[�^���C���[
//=======================================
#ifndef __GRAVISBELL_I_DATA_LAYER_H__
#define __GRAVISBELL_I_DATA_LAYER_H__

#include"Common/ErrorCode.h"
#include"Common/IODataStruct.h"

#include"IOutputLayer.h"


namespace Gravisbell {
namespace NeuralNetwork {


	/** ���̓f�[�^���C���[ */
	class IDataLayer
	{
	public:
		/** �R���X�g���N�^ */
		IDataLayer(){}
		/** �f�X�g���N�^ */
		virtual ~IDataLayer(){}

	public:
		/** �f�[�^�̍\�������擾���� */
		virtual IODataStruct GetDataStruct()const = 0;

		/** �f�[�^�̃o�b�t�@�T�C�Y���擾����.
			@return �f�[�^�̃o�b�t�@�T�C�Y.�g�p����float�^�z��̗v�f��. */
		virtual unsigned int GetBufferCount()const = 0;

		/** �f�[�^��ǉ�����.
			@param	lpData	�f�[�^��g�̔z��. [GetBufferSize()�̖߂�l]�̗v�f�����K�v. */
		virtual ErrorCode AddData(const float lpData[]) = 0;

		/** �f�[�^�����擾���� */
		virtual unsigned int GetDataCount()const = 0;
		/** �f�[�^��ԍ��w��Ŏ擾����.
			@param num		�擾����ԍ�
			@param o_LpData �f�[�^�̊i�[��z��. [GetBufferSize()�̖߂�l]�̗v�f�����K�v. */
		virtual ErrorCode GetDataByNum(unsigned int num, float o_lpData[])const = 0;
		/** �f�[�^��ԍ��w��ŏ������� */
		virtual ErrorCode EraseDataByNum(unsigned int num) = 0;

		/** �f�[�^��S��������.
			@return	���������ꍇ0 */
		virtual ErrorCode ClearData() = 0;

		/** �o�b�`�����f�[�^�ԍ����X�g��ݒ肷��.
			�ݒ肳�ꂽ�l������GetDInputBuffer(),GetOutputBuffer()�̖߂�l�����肷��.
			@param i_lpBatchDataNoList	�ݒ肷��f�[�^�ԍ����X�g. [GetBatchSize()�̖߂�l]�̗v�f�����K�v */
		virtual ErrorCode SetBatchDataNoList(const unsigned int i_lpBatchDataNoList[]) = 0;
	};

}	// NeuralNetwork
}	// Gravisbell

#endif