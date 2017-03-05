//=======================================
// �o�b�`�����f�[�^�ԍ����X�g�����N���X
//=======================================
#ifndef __GRAVISBELL_I_BATCH_DATA_NO_LIST_GENERATOR_H__
#define __GRAVISBELL_I_BATCH_DATA_NO_LIST_GENERATOR_H__

#include"Common/ErrorCode.h"
#include"Common/IODataStruct.h"

#include"IOutputLayer.h"

namespace Gravisbell {
namespace NeuralNetwork {

	/** ���̓f�[�^���C���[ */
	class IBatchDataNoListGenerator
	{
	public:
		/** �R���X�g���N�^ */
		IBatchDataNoListGenerator(){}
		/** �f�X�g���N�^ */
		virtual ~IBatchDataNoListGenerator(){}


	public:
		/** ���Z�O���������s����.
			@param dataCount	���f�[�^��
			@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
			NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
			���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
		virtual ErrorCode PreProcess(unsigned int dataCount, unsigned int batchSize) = 0;

		/** �w�K���[�v�̏���������.�f�[�^�Z�b�g�̊w�K�J�n�O�Ɏ��s����
			���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
		virtual ErrorCode PreProcessLearnLoop() = 0;


	public:
		/** �f�[�^�����擾���� */
		virtual unsigned int GetDataCount()const = 0;

		/** �o�b�`�T�C�Y���擾����.
			@return �����ɉ��Z���s���o�b�`�̃T�C�Y */
		virtual unsigned int GetBatchSize()const = 0;


	public:
		/** �f�[�^�ԍ����X�g�����擾����.
			@return	�f�[�^�ԍ����X�g�̑��� = �f�[�^�� / �o�b�`�T�C�Y (�[���؂�グ)���Ԃ� */
		virtual unsigned int GetBatchDataNoListCount()const = 0;

		/** �f�[�^�ԍ����X�g���擾����.
			@param	no	�擾����f�[�^�ԍ����X�g�̔ԍ�. 0 <= n < GetBatchDataNoListCount() �܂ł͈̔�.
			@return	�f�[�^�ԍ����X�g�̔z�񂪕ς���. [GetBatchSize()]�̗v�f�� */
		virtual const unsigned int* GetBatchDataNoListByNum(unsigned int no)const = 0;
	};

}	// NeuralNetwork
}	// Gravisbell

#endif