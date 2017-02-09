//=======================================
// �f�[�^���C���[
//=======================================
#ifndef __I_DATA_LAYER_H__
#define __I_DATA_LAYER_H__

#include"LayerErrorCode.h"
#include"IOutputLayer.h"
#include"IODataStruct.h"

namespace CustomDeepNNLibrary
{
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
			@param	lpData	�f�[�^��g�̔z��. GetBufferSize()�̖߂�l�̗v�f�����K�v. */
		virtual ELayerErrorCode AddData(const float lpData[]) = 0;

		/** �f�[�^�����擾���� */
		virtual unsigned int GetDataCount()const = 0;
		/** �f�[�^��ԍ��w��Ŏ擾����.
			@param num		�擾����ԍ�
			@param o_LpData �f�[�^�̊i�[��z��. GetBufferSize()�̖߂�l�̗v�f�����K�v. */
		virtual ELayerErrorCode GetDataByNum(unsigned int num, float o_lpData[])const = 0;
		/** �g�p�f�[�^�̐؂�ւ� */
		virtual ELayerErrorCode ChangeUseDataByNum(unsigned int num) = 0;
		/** �f�[�^��ԍ��w��ŏ������� */
		virtual ELayerErrorCode EraseDataByNum(unsigned int num) = 0;

		/** �f�[�^��S��������.
			@return	���������ꍇ0 */
		virtual ELayerErrorCode ClearData() = 0;
	};
}

#endif