// IODataLayer.cpp : DLL �A�v���P�[�V�����p�ɃG�N�X�|�[�g�����֐����`���܂��B
//

#include "stdafx.h"
#include "IODataLayer.h"

#include<vector>
#include<list>

// UUID�֘A�p
#include<rpc.h>
#pragma comment(lib, "Rpcrt4.lib")

namespace Gravisbell {
namespace Layer {
namespace IOData {

	class IODataLayerCPU : public IIODataLayer
	{
	private:
		GUID guid;	/**< ����ID */
		Gravisbell::IODataStruct ioDataStruct;	/**< �f�[�^�\�� */

		std::vector<F32*> lpBufferList;
		std::vector<std::vector<F32>> lpDInputBuffer;	/**< �덷�����̕ۑ��o�b�t�@ */

		U32 batchSize;	/**< �o�b�`�����T�C�Y */
		const U32* lpBatchDataNoList;	/**< �o�b�`�����f�[�^�ԍ����X�g */

		std::vector<F32*> lpBatchDataPointer;			/**< �o�b�`�����f�[�^�̔z��擪�A�h���X���X�g */
		std::vector<F32*> lpBatchDInputBufferPointer;	/**< �o�b�`�������͌덷�����̔z��擱�A�h���X���X�g */

	public:
		/** �R���X�g���N�^ */
		IODataLayerCPU(GUID guid, Gravisbell::IODataStruct ioDataStruct)
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
		U32 GetLayerKind()const
		{
			return ELayerKind::LAYER_KIND_CPU | ELayerKind::LAYER_KIND_SINGLE_INPUT | ELayerKind::LAYER_KIND_SINGLE_OUTPUT | ELayerKind::LAYER_KIND_DATA;
		}

		/** ���C���[�ŗL��GUID���擾���� */
		Gravisbell::ErrorCode GetGUID(GUID& o_guid)const
		{
			o_guid = this->guid;

			return Gravisbell::ErrorCode::ERROR_CODE_NONE;
		}

		/** ���C���[��ʎ��ʃR�[�h���擾����.
			@param o_layerCode	�i�[��o�b�t�@
			@return ���������ꍇ0 */
		Gravisbell::ErrorCode GetLayerCode(GUID& o_layerCode)const
		{
			return Gravisbell::Layer::IOData::GetLayerCode(o_layerCode);
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
			@return �f�[�^�̃o�b�t�@�T�C�Y.�g�p����F32�^�z��̗v�f��. */
		U32 GetBufferCount()const
		{
			return this->ioDataStruct.ch * this->ioDataStruct.x * this->ioDataStruct.y * this->ioDataStruct.z;
		}

		/** �f�[�^��ǉ�����.
			@param	lpData	�f�[�^��g�̔z��. GetBufferSize()�̖߂�l�̗v�f�����K�v.
			@return	�ǉ����ꂽ�ۂ̃f�[�^�Ǘ��ԍ�. ���s�����ꍇ�͕��̒l. */
		Gravisbell::ErrorCode AddData(const F32 lpData[])
		{
			if(lpData == NULL)
				return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

			// �o�b�t�@�m��
			F32* lpBuffer = new F32[this->GetBufferCount()];
			if(lpBuffer == NULL)
				return ErrorCode::ERROR_CODE_COMMON_ALLOCATION_MEMORY;

			// �R�s�[
			memcpy(lpBuffer, lpData, sizeof(F32)*this->GetBufferCount());

			// ���X�g�ɒǉ�
			lpBufferList.push_back(lpBuffer);

			return Gravisbell::ErrorCode::ERROR_CODE_NONE;
		}

		/** �f�[�^�����擾���� */
		U32 GetDataCount()const
		{
			return this->lpBufferList.size();
		}
		/** �f�[�^��ԍ��w��Ŏ擾����.
			@param num		�擾����ԍ�
			@param o_lpBufferList �f�[�^�̊i�[��z��. GetBufferSize()�̖߂�l�̗v�f�����K�v.
			@return ���������ꍇ0 */
		Gravisbell::ErrorCode GetDataByNum(U32 num, F32 o_lpBufferList[])const
		{
			if(num >= this->lpBufferList.size())
				return ErrorCode::ERROR_CODE_COMMON_OUT_OF_ARRAYRANGE;

			if(o_lpBufferList == NULL)
				return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

			for(U32 i=0; i<this->GetBufferCount(); i++)
			{
				o_lpBufferList[i] = this->lpBufferList[num][i];
			}

			return Gravisbell::ErrorCode::ERROR_CODE_NONE;
		}
		/** �f�[�^��ԍ��w��ŏ������� */
		Gravisbell::ErrorCode EraseDataByNum(U32 num)
		{
			if(num >= this->lpBufferList.size())
				return ErrorCode::ERROR_CODE_COMMON_OUT_OF_ARRAYRANGE;

			// �ԍ��̏ꏊ�܂ňړ�
			auto it = this->lpBufferList.begin();
			for(U32 i=0; i<num; i++)
				it++;

			// �폜
			if(*it != NULL)
				delete *it;
			this->lpBufferList.erase(it);

			return Gravisbell::ErrorCode::ERROR_CODE_NONE;
		}

		/** �f�[�^��S��������.
			@return	���������ꍇ0 */
		Gravisbell::ErrorCode ClearData()
		{
			for(U32 i=0; i<lpBufferList.size(); i++)
			{
				if(lpBufferList[i] != NULL)
					delete lpBufferList[i];
			}

			return Gravisbell::ErrorCode::ERROR_CODE_NONE;
		}

		/** �o�b�`�����f�[�^�ԍ����X�g��ݒ肷��.
			�ݒ肳�ꂽ�l������GetDInputBuffer(),GetOutputBuffer()�̖߂�l�����肷��.
			@param i_lpBatchDataNoList	�ݒ肷��f�[�^�ԍ����X�g. [GetBatchSize()�̖߂�l]�̗v�f�����K�v */
		Gravisbell::ErrorCode SetBatchDataNoList(const U32 i_lpBatchDataNoList[])
		{
			this->lpBatchDataNoList = i_lpBatchDataNoList;

			for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
			{
				if(this->lpBatchDataNoList[batchNum] > this->lpBufferList.size())
					return Gravisbell::ErrorCode::ERROR_CODE_COMMON_OUT_OF_ARRAYRANGE;

				this->lpBatchDataPointer[batchNum] = this->lpBufferList[this->lpBatchDataNoList[batchNum]];
			}

			return Gravisbell::ErrorCode::ERROR_CODE_NONE;
		}



		//==============================
		// ���C���[���ʌn
		//==============================
	public:
		/** ���Z�O���������s����.(�w�K�p)
			@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
			NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
			���s�����ꍇ��PreProcessLearnLoop�ȍ~�̏����͎��s�s��. */
		Gravisbell::ErrorCode PreProcessLearn(U32 batchSize)
		{
			// �o�b�`�����f�[�^�z��̏�����
			this->batchSize = batchSize;
			this->lpBatchDataPointer.resize(batchSize);

			// �덷�����f�[�^�z��̏�����
			this->lpDInputBuffer.resize(batchSize);
			this->lpBatchDInputBufferPointer.resize(batchSize);
			for(U32 i=0; i<this->lpDInputBuffer.size(); i++)
			{
				this->lpDInputBuffer[i].resize(this->GetInputBufferCount());
				this->lpBatchDInputBufferPointer[i] = &this->lpDInputBuffer[i][0];
			}

			return Gravisbell::ErrorCode::ERROR_CODE_NONE;
		}

		/** ���Z�O���������s����.(���Z�p)
			@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
			NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
			���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
		Gravisbell::ErrorCode PreProcessCalculate(U32 batchSize)
		{
			// �o�b�`�����f�[�^�z��̏�����
			this->batchSize = batchSize;
			this->lpBatchDataPointer.resize(batchSize);

			return Gravisbell::ErrorCode::ERROR_CODE_NONE;
		}

		/** �w�K���[�v�̏���������.�f�[�^�Z�b�g�̊w�K�J�n�O�Ɏ��s����
			���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
		Gravisbell::ErrorCode PreProcessLearnLoop(const SettingData::Standard::IData& config)
		{
			return Gravisbell::ErrorCode::ERROR_CODE_NONE;
		}

		/** �o�b�`�T�C�Y���擾����.
			@return �����ɉ��Z���s���o�b�`�̃T�C�Y */
		U32 GetBatchSize()const
		{
			return this->batchSize;
		}


		//==============================
		// ���͌n
		//==============================
	public:
		/** �w�K�덷���v�Z����.
			@param i_lppInputBuffer	���̓f�[�^�o�b�t�@. [GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]�̗v�f�����K�v */
		Gravisbell::ErrorCode CalculateLearnError(Gravisbell::CONST_BATCH_BUFFER_POINTER i_lppInputBuffer)
		{
			U32 inputBufferCount = this->GetInputBufferCount();

			for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
			{
				for(U32 inputNum=0; inputNum<inputBufferCount; inputNum++)
				{
					this->lpDInputBuffer[batchNum][inputNum] = this->lpBatchDataPointer[batchNum][inputNum] - i_lppInputBuffer[batchNum][inputNum];
				}
			}

			return Gravisbell::ErrorCode::ERROR_CODE_NONE;
		}

	public:
		/** ���̓f�[�^�\�����擾����.
			@return	���̓f�[�^�\�� */
		IODataStruct GetInputDataStruct()const
		{
			return this->GetDataStruct();
		}

		/** ���̓o�b�t�@�����擾����. byte���ł͖����f�[�^�̐��Ȃ̂Œ��� */
		U32 GetInputBufferCount()const
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
		Gravisbell::ErrorCode GetDInputBuffer(BATCH_BUFFER_POINTER o_lpDInputBuffer)const
		{
			if(o_lpDInputBuffer == NULL)
				return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;
			
			const U32 batchSize = this->GetBatchSize();
			const U32 inputBufferCount = this->GetOutputBufferCount();

			for(U32 batchNum=0; batchNum<batchSize; batchNum++)
			{
				memcpy(o_lpDInputBuffer[batchNum], this->lpBatchDataPointer[batchNum], sizeof(F32)*inputBufferCount);
			}

			return Gravisbell::ErrorCode::ERROR_CODE_NONE;
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
		U32 GetOutputBufferCount()const
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
		Gravisbell::ErrorCode GetOutputBuffer(BATCH_BUFFER_POINTER o_lpOutputBuffer)const
		{
			if(o_lpOutputBuffer == NULL)
				return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

			const U32 batchSize = this->GetBatchSize();
			const U32 outputBufferCount = this->GetOutputBufferCount();

			for(U32 batchNum=0; batchNum<batchSize; batchNum++)
			{
				memcpy(o_lpOutputBuffer[batchNum], this->lpBatchDataPointer[batchNum], sizeof(F32)*outputBufferCount);
			}

			return Gravisbell::ErrorCode::ERROR_CODE_NONE;
		}
	};

	/** ���͐M���f�[�^���C���[���쐬����.GUID�͎������蓖��.CPU����
		@param bufferSize	�o�b�t�@�̃T�C�Y.��F32�^�z��̗v�f��.
		@return	���͐M���f�[�^���C���[�̃A�h���X */
	extern IODataLayer_API Gravisbell::Layer::IOData::IIODataLayer* CreateIODataLayerCPU(Gravisbell::IODataStruct ioDataStruct)
	{
		UUID uuid;
		::UuidCreate(&uuid);

		return CreateIODataLayerCPUwithGUID(uuid, ioDataStruct);
	}
	/** ���͐M���f�[�^���C���[���쐬����.CPU����
		@param guid			���C���[��GUID.
		@param bufferSize	�o�b�t�@�̃T�C�Y.��F32�^�z��̗v�f��.
		@return	���͐M���f�[�^���C���[�̃A�h���X */
	extern IODataLayer_API Gravisbell::Layer::IOData::IIODataLayer* CreateIODataLayerCPUwithGUID(GUID guid, Gravisbell::IODataStruct ioDataStruct)
	{
		return new Gravisbell::Layer::IOData::IODataLayerCPU(guid, ioDataStruct);
	}

}	// IOData
}	// Layer
}	// Gravisbell
