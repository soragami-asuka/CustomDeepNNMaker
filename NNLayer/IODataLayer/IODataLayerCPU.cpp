// IODataLayer.cpp : DLL �A�v���P�[�V�����p�ɃG�N�X�|�[�g�����֐����`���܂��B
//

#include "stdafx.h"
#include "IODataLayer.h"

#include<vector>
#include<list>

#include<rpc.h>
#pragma comment(lib, "Rpcrt4.lib")

namespace CustomDeepNNLibrary
{
	class IODataLayerCPU : public CustomDeepNNLibrary::IIODataLayer
	{
	private:
		GUID guid;	/**< ����ID */
		CustomDeepNNLibrary::IODataStruct ioDataStruct;	/**< �f�[�^�\�� */

		std::vector<float*> lpBufferList;
		std::vector<std::vector<float>> lpDInputBuffer;	/**< �덷�����̕ۑ��o�b�t�@ */

		unsigned int batchSize;	/**< �o�b�`�����T�C�Y */
		const unsigned int* lpBatchDataNoList;	/**< �o�b�`�����f�[�^�ԍ����X�g */

		std::vector<float*> lpBatchDataPointer;			/**< �o�b�`�����f�[�^�̔z��擪�A�h���X���X�g */
		std::vector<float*> lpBatchDInputBufferPointer;	/**< �o�b�`�������͌덷�����̔z��擱�A�h���X���X�g */

	public:
		/** �R���X�g���N�^ */
		IODataLayerCPU(GUID guid, CustomDeepNNLibrary::IODataStruct ioDataStruct)
			:	guid				(guid)
			,	ioDataStruct		(ioDataStruct)
			,	lpBatchDataNoList	(NULL)
		{
		}
		/** �f�X�g���N�^ */
		virtual ~IODataLayerCPU()
		{
			this->ClearData();
		}


		//==============================
		// ���C���[���ʌn
		//==============================
	public:
		/** ���C���[��ʂ̎擾 */
		unsigned int GetLayerKind()const
		{
			return ELayerKind::LAYER_KIND_CPU | ELayerKind::LAYER_KIND_SINGLE_INPUT | ELayerKind::LAYER_KIND_SINGLE_OUTPUT | ELayerKind::LAYER_KIND_DATA;
		}

		/** ���C���[�ŗL��GUID���擾���� */
		ELayerErrorCode GetGUID(GUID& o_guid)const
		{
			o_guid = this->guid;

			return LAYER_ERROR_NONE;
		}

		/** ���C���[��ʎ��ʃR�[�h���擾����.
			@param o_layerCode	�i�[��o�b�t�@
			@return ���������ꍇ0 */
		ELayerErrorCode GetLayerCode(GUID& o_layerCode)const
		{
			// {6E99D406-B931-4DE0-AC3A-48A35E129820}
			o_layerCode = { 0x6e99d406, 0xb931, 0x4de0, { 0xac, 0x3a, 0x48, 0xa3, 0x5e, 0x12, 0x98, 0x20 } };

			return LAYER_ERROR_NONE;
		}

		//==============================
		// �f�[�^�Ǘ��n
		//==============================
	public:
		/** �f�[�^�̍\�������擾���� */
		IODataStruct GetDataStruct()const
		{
			return this->ioDataStruct;
		}

		/** �f�[�^�̃o�b�t�@�T�C�Y���擾����.
			@return �f�[�^�̃o�b�t�@�T�C�Y.�g�p����float�^�z��̗v�f��. */
		unsigned int GetBufferCount()const
		{
			return this->ioDataStruct.ch * this->ioDataStruct.x * this->ioDataStruct.y * this->ioDataStruct.z;
		}

		/** �f�[�^��ǉ�����.
			@param	lpData	�f�[�^��g�̔z��. GetBufferSize()�̖߂�l�̗v�f�����K�v.
			@return	�ǉ����ꂽ�ۂ̃f�[�^�Ǘ��ԍ�. ���s�����ꍇ�͕��̒l. */
		ELayerErrorCode AddData(const float lpData[])
		{
			if(lpData == NULL)
				return LAYER_ERROR_COMMON_NULL_REFERENCE;

			// �o�b�t�@�m��
			float* lpBuffer = new float[this->GetBufferCount()];
			if(lpBuffer == NULL)
				return LAYER_ERROR_COMMON_ALLOCATION_MEMORY;

			// �R�s�[
			memcpy(lpBuffer, lpData, sizeof(float)*this->GetBufferCount());

			// ���X�g�ɒǉ�
			lpBufferList.push_back(lpBuffer);

			return LAYER_ERROR_NONE;
		}

		/** �f�[�^�����擾���� */
		unsigned int GetDataCount()const
		{
			return this->lpBufferList.size();
		}
		/** �f�[�^��ԍ��w��Ŏ擾����.
			@param num		�擾����ԍ�
			@param o_lpBufferList �f�[�^�̊i�[��z��. GetBufferSize()�̖߂�l�̗v�f�����K�v.
			@return ���������ꍇ0 */
		ELayerErrorCode GetDataByNum(unsigned int num, float o_lpBufferList[])const
		{
			if(num >= this->lpBufferList.size())
				return LAYER_ERROR_COMMON_OUT_OF_ARRAYRANGE;

			if(o_lpBufferList == NULL)
				return LAYER_ERROR_COMMON_NULL_REFERENCE;

			for(unsigned int i=0; i<this->GetBufferCount(); i++)
			{
				o_lpBufferList[i] = this->lpBufferList[num][i];
			}

			return LAYER_ERROR_NONE;
		}
		/** �f�[�^��ԍ��w��ŏ������� */
		ELayerErrorCode EraseDataByNum(unsigned int num)
		{
			if(num >= this->lpBufferList.size())
				return LAYER_ERROR_COMMON_OUT_OF_ARRAYRANGE;

			// �ԍ��̏ꏊ�܂ňړ�
			auto it = this->lpBufferList.begin();
			for(unsigned int i=0; i<num; i++)
				it++;

			// �폜
			if(*it != NULL)
				delete *it;
			this->lpBufferList.erase(it);

			return LAYER_ERROR_NONE;
		}

		/** �f�[�^��S��������.
			@return	���������ꍇ0 */
		ELayerErrorCode ClearData()
		{
			for(unsigned int i=0; i<lpBufferList.size(); i++)
			{
				if(lpBufferList[i] != NULL)
					delete lpBufferList[i];
			}

			return LAYER_ERROR_NONE;
		}

		/** �o�b�`�����f�[�^�ԍ����X�g��ݒ肷��.
			�ݒ肳�ꂽ�l������GetDInputBuffer(),GetOutputBuffer()�̖߂�l�����肷��.
			@param i_lpBatchDataNoList	�ݒ肷��f�[�^�ԍ����X�g. [GetBatchSize()�̖߂�l]�̗v�f�����K�v */
		ELayerErrorCode SetBatchDataNoList(const unsigned int i_lpBatchDataNoList[])
		{
			this->lpBatchDataNoList = i_lpBatchDataNoList;

			for(unsigned int batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
			{
				if(this->lpBatchDataNoList[batchNum] > this->lpBufferList.size())
					return ELayerErrorCode::LAYER_ERROR_COMMON_OUT_OF_ARRAYRANGE;

				this->lpBatchDataPointer[batchNum] = this->lpBufferList[this->lpBatchDataNoList[batchNum]];
			}

			return ELayerErrorCode::LAYER_ERROR_NONE;
		}



		//==============================
		// ���C���[���ʌn
		//==============================
	public:
		/** ���Z�O���������s����.(�w�K�p)
			@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
			NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
			���s�����ꍇ��PreProcessLearnLoop�ȍ~�̏����͎��s�s��. */
		ELayerErrorCode PreProcessLearn(unsigned int batchSize)
		{
			// �o�b�`�����f�[�^�z��̏�����
			this->batchSize = batchSize;
			this->lpBatchDataPointer.resize(batchSize);

			// �덷�����f�[�^�z��̏�����
			this->lpDInputBuffer.resize(batchSize);
			this->lpBatchDInputBufferPointer.resize(batchSize);
			for(unsigned int i=0; i<this->lpDInputBuffer.size(); i++)
			{
				this->lpDInputBuffer[i].resize(this->GetInputBufferCount());
				this->lpBatchDInputBufferPointer[i] = &this->lpDInputBuffer[i][0];
			}

			return ELayerErrorCode::LAYER_ERROR_NONE;
		}

		/** ���Z�O���������s����.(���Z�p)
			@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
			NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
			���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
		ELayerErrorCode PreProcessCalculate(unsigned int batchSize)
		{
			// �o�b�`�����f�[�^�z��̏�����
			this->batchSize = batchSize;
			this->lpBatchDataPointer.resize(batchSize);

			return ELayerErrorCode::LAYER_ERROR_NONE;
		}

		/** �w�K���[�v�̏���������.�f�[�^�Z�b�g�̊w�K�J�n�O�Ɏ��s����
			���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
		ELayerErrorCode PreProcessLearnLoop(const INNLayerConfig& config)
		{
			return ELayerErrorCode::LAYER_ERROR_NONE;
		}

		/** �o�b�`�T�C�Y���擾����.
			@return �����ɉ��Z���s���o�b�`�̃T�C�Y */
		unsigned int GetBatchSize()const
		{
			return this->batchSize;
		}


		//==============================
		// ���͌n
		//==============================
	public:
		/** �w�K�덷���v�Z����.
			@param i_lppInputBuffer	���̓f�[�^�o�b�t�@. [GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]�̗v�f�����K�v
			���O�̌v�Z���ʂ��g�p���� */
		ELayerErrorCode CalculateLearnError(const float** i_lppInputBuffer)
		{
			unsigned int inputBufferCount = this->GetInputBufferCount();

			for(unsigned int batchNum=0; batchNum<this->batchSize; batchNum++)
			{
				for(unsigned int inputNum=0; inputNum<inputBufferCount; inputNum++)
				{
					this->lpDInputBuffer[batchNum][inputNum] = this->lpBatchDataPointer[batchNum][inputNum] - i_lppInputBuffer[batchNum][inputNum];
				}
			}

			return LAYER_ERROR_NONE;
		}

	public:
		/** ���̓f�[�^�\�����擾����.
			@return	���̓f�[�^�\�� */
		const IODataStruct GetInputDataStruct()const
		{
			return this->GetDataStruct();
		}
		/** ���̓f�[�^�\�����擾����
			@param	o_inputDataStruct	���̓f�[�^�\���̊i�[��
			@return	���������ꍇ0 */
		ELayerErrorCode GetInputDataStruct(IODataStruct& o_inputDataStruct)const
		{
			o_inputDataStruct = this->GetDataStruct();

			return ELayerErrorCode::LAYER_ERROR_NONE;
		}

		/** ���̓o�b�t�@�����擾����. byte���ł͖����f�[�^�̐��Ȃ̂Œ��� */
		unsigned int GetInputBufferCount()const
		{
			return this->GetBufferCount();
		}

		/** �w�K�������擾����.
			�z��̗v�f����GetInputBufferCount�̖߂�l.
			@return	�덷�����z��̐擪�|�C���^ */
		CONST_BATCH_BUFFER_POINTER GetDInputBuffer()const
		{
			return &this->lpBatchDInputBufferPointer[0];
		}
		/** �w�K�������擾����.
			@param lpDOutputBuffer	�w�K�������i�[����z��.[GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]�̔z�񂪕K�v */
		ELayerErrorCode GetDInputBuffer(BATCH_BUFFER_POINTER o_lpDInputBuffer)const
		{
			if(o_lpDInputBuffer == NULL)
				return LAYER_ERROR_COMMON_NULL_REFERENCE;
			
			const unsigned int batchSize = this->GetBatchSize();
			const unsigned int inputBufferCount = this->GetOutputBufferCount();

			for(unsigned int batchNum=0; batchNum<batchSize; batchNum++)
			{
				memcpy(o_lpDInputBuffer[batchNum], this->lpBatchDataPointer[batchNum], sizeof(float)*inputBufferCount);
			}

			return LAYER_ERROR_NONE;

			return LAYER_ERROR_NONE;
		}


		//==============================
		// �o�͌n
		//==============================
	public:
		/** �o�̓f�[�^�\�����擾���� */
		IODataStruct GetOutputDataStruct()const
		{
			return this->GetDataStruct();
		}

		/** �o�̓o�b�t�@�����擾����. byte���ł͖����f�[�^�̐��Ȃ̂Œ��� */
		unsigned int GetOutputBufferCount()const
		{
			return this->GetBufferCount();
		}

		/** �o�̓f�[�^�o�b�t�@���擾����.
			�z��̗v�f����GetOutputBufferCount�̖߂�l.
			@return �o�̓f�[�^�z��̐擪�|�C���^ */
		CONST_BATCH_BUFFER_POINTER GetOutputBuffer()const
		{
			return &this->lpBatchDataPointer[0];
		}
		/** �o�̓f�[�^�o�b�t�@���擾����.
			@param o_lpOutputBuffer	�o�̓f�[�^�i�[��z��. [GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v
			@return ���������ꍇ0 */
		ELayerErrorCode GetOutputBuffer(BATCH_BUFFER_POINTER o_lpOutputBuffer)const
		{
			if(o_lpOutputBuffer == NULL)
				return LAYER_ERROR_COMMON_NULL_REFERENCE;

			const unsigned int batchSize = this->GetBatchSize();
			const unsigned int outputBufferCount = this->GetOutputBufferCount();

			for(unsigned int batchNum=0; batchNum<batchSize; batchNum++)
			{
				memcpy(o_lpOutputBuffer[batchNum], this->lpBatchDataPointer[batchNum], sizeof(float)*outputBufferCount);
			}

			return LAYER_ERROR_NONE;
		}
	};
}

/** ���͐M���f�[�^���C���[���쐬����.GUID�͎������蓖��.CPU����
	@param bufferSize	�o�b�t�@�̃T�C�Y.��float�^�z��̗v�f��.
	@return	���͐M���f�[�^���C���[�̃A�h���X */
extern "C" IODataLayer_API CustomDeepNNLibrary::IIODataLayer* CreateIODataLayerCPU(CustomDeepNNLibrary::IODataStruct ioDataStruct)
{
	UUID uuid;
	::UuidCreate(&uuid);

	return CreateIODataLayerCPUwithGUID(uuid, ioDataStruct);
}
/** ���͐M���f�[�^���C���[���쐬����.CPU����
	@param guid			���C���[��GUID.
	@param bufferSize	�o�b�t�@�̃T�C�Y.��float�^�z��̗v�f��.
	@return	���͐M���f�[�^���C���[�̃A�h���X */
extern "C" IODataLayer_API CustomDeepNNLibrary::IIODataLayer* CreateIODataLayerCPUwithGUID(GUID guid, CustomDeepNNLibrary::IODataStruct ioDataStruct)
{
	return new CustomDeepNNLibrary::IODataLayerCPU(guid, ioDataStruct);
}