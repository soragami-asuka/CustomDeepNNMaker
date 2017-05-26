// BatchDataNoListGenerator.cpp : DLL �A�v���P�[�V�����p�ɃG�N�X�|�[�g�����֐����`���܂��B
//

#include "stdafx.h"
#include "BatchDataNoListGenerator.h"

#include<vector>
#include<algorithm>
#include <random>

using namespace Gravisbell;

namespace Gravisbell {
namespace Common {

	class BatchDataNoListGenerator : public Gravisbell::Common::IBatchDataNoListGenerator
	{
	private:
		unsigned int dataCount;
		unsigned int batchSize;

		std::vector<unsigned int> lpAllDataNoList;		// �S�f�[�^�ԍ��̃����_���z��
		std::vector<unsigned int>::iterator it_addDataBegin;		// �[�����̊J�n�C�e���[�^

		std::random_device seed_gen;
		std::mt19937 random_generator;

	public:
		/** �R���X�g���N�^ */
		BatchDataNoListGenerator()
			:	IBatchDataNoListGenerator	()
			,	seed_gen					()
			,	random_generator			(seed_gen())
		{
		}
		/** �f�X�g���N�^ */
		virtual ~BatchDataNoListGenerator()
		{
		}


	public:
		/** ���Z�O���������s����.
			@param dataCount	���f�[�^��
			@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
			NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
			���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
		Gravisbell::ErrorCode PreProcess(unsigned int dataCount, unsigned int batchSize)
		{
			this->dataCount = dataCount;
			this->batchSize = batchSize;

			// �z�񏉊���
			this->lpAllDataNoList.resize( (dataCount + (batchSize-1)) / batchSize * batchSize );
			for(unsigned int i=0; i<dataCount; i++)
			{
				this->lpAllDataNoList[i] = i;
			}

			// �[���̊J�n�C�e���[�^���擾
			this->it_addDataBegin = this->lpAllDataNoList.begin();
			for(unsigned int i=0; i<dataCount; i++)
				this->it_addDataBegin++;

			// �[��������������
			U32 addDataCount = (U32)this->lpAllDataNoList.size() - this->dataCount;
			for(unsigned int i=0; i<addDataCount; i++)
			{
				this->lpAllDataNoList[this->dataCount + i] = this->lpAllDataNoList[i];
			}

			return Gravisbell::ErrorCode::ERROR_CODE_NONE;
		}

		/** �w�K���[�v�̏���������.�f�[�^�Z�b�g�̊w�K�J�n�O�Ɏ��s����
			���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
		Gravisbell::ErrorCode PreProcessLearnLoop()
		{
			// �V���b�t��
			std::shuffle(this->lpAllDataNoList.begin(), this->it_addDataBegin, random_generator);

			//// �[��������������
			//U32 addDataCount = (U32)this->lpAllDataNoList.size() - this->dataCount;
			//for(unsigned int i=0; i<addDataCount; i++)
			//{
			//	this->lpAllDataNoList[this->dataCount + i] = this->lpAllDataNoList[i];
			//}

			return Gravisbell::ErrorCode::ERROR_CODE_NONE;
		}

	public:
		/** �f�[�^�����擾���� */
		unsigned int GetDataCount()const
		{
			return this->dataCount;
		}

		/** �o�b�`�T�C�Y���擾����.
			@return �����ɉ��Z���s���o�b�`�̃T�C�Y */
		unsigned int GetBatchSize()const
		{
			return this->batchSize;
		}


	public:
		/** �f�[�^�ԍ����X�g�����擾����.
			@return	�f�[�^�ԍ����X�g�̑��� = �f�[�^�� / �o�b�`�T�C�Y (�[���؂�グ)���Ԃ� */
		unsigned int GetBatchDataNoListCount()const
		{
			return (U32)this->lpAllDataNoList.size() / this->GetBatchSize();
		}

		/** �f�[�^�ԍ����X�g���擾����.
			@param	no	�擾����f�[�^�ԍ����X�g�̔ԍ�. 0 <= n < GetBatchDataNoListCount() �܂ł͈̔�.
			@return	�f�[�^�ԍ����X�g�̔z�񂪕ς���. [GetBatchSize()]�̗v�f�� */
		const unsigned int* GetBatchDataNoListByNum(unsigned int no)const
		{
			return &this->lpAllDataNoList[no * this->batchSize];
		}
	};


	/** �o�b�`�����f�[�^�ԍ����X�g�����N���X���쐬����. */
	extern BatchDataNoListGenerator_API Gravisbell::Common::IBatchDataNoListGenerator* CreateBatchDataNoListGenerator()
	{
		return new BatchDataNoListGenerator();
	}

}	// Common
}	// Gravisbell
