//===================================
// ���o�̓f�[�^���Ǘ�����N���X
// GPU����
// device�������m�ی^
//===================================


#include "stdafx.h"
#include "IODataLayerGPU_base.cuh"


#include<vector>
#include<list>
#include<algorithm>

// UUID�֘A�p
#include<boost/uuid/uuid_generators.hpp>


namespace Gravisbell {
namespace Layer {
namespace IOData {

	class IODataLayerGPU_device : public IODataLayerGPU_base
	{
	private:
		std::vector<F32*> lpBufferList;


	public:
		/** �R���X�g���N�^ */
		IODataLayerGPU_device(Gravisbell::GUID guid, Gravisbell::IODataStruct ioDataStruct)
			:	IODataLayerGPU_base	(guid, ioDataStruct)
		{
		}
		/** �f�X�g���N�^ */
		virtual ~IODataLayerGPU_device()
		{
			this->ClearData();
		}


		//==============================
		// �f�[�^�Ǘ��n
		//==============================
	public:
		/** �f�[�^��ǉ�����.
			@param	lpData	�f�[�^��g�̔z��. GetBufferSize()�̖߂�l�̗v�f�����K�v.
			@return	�ǉ����ꂽ�ۂ̃f�[�^�Ǘ��ԍ�. ���s�����ꍇ�͕��̒l. */
		Gravisbell::ErrorCode AddData(const F32 lpData[])
		{
			if(lpData == NULL)
				return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

			// �o�b�t�@�m��
			F32* lpBuffer = NULL;
			cudaError_t err = cudaMalloc((void**)&lpBuffer, sizeof(F32) * this->GetBufferCount());
			if(err != 0)
				return ErrorCode::ERROR_CODE_COMMON_ALLOCATION_MEMORY;

			// �R�s�[
			cudaMemcpy((void*)lpBuffer, lpData, sizeof(F32)*this->GetBufferCount(), cudaMemcpyHostToDevice);

			// ���X�g�ɒǉ�
			lpBufferList.push_back(lpBuffer);

			return Gravisbell::ErrorCode::ERROR_CODE_NONE;
		}

		/** �f�[�^�����擾���� */
		U32 GetDataCount()const
		{
			return (U32)this->lpBufferList.size();
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

			cudaMemcpy(o_lpBufferList, this->lpBufferList[num], sizeof(F32)*this->GetOutputBufferCount(), cudaMemcpyDeviceToHost);

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
				cudaFree(*it);
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
					cudaFree(lpBufferList[i]);
			}
			this->lpBufferList.clear();

			return Gravisbell::ErrorCode::ERROR_CODE_NONE;
		}

		/** �o�b�`�����f�[�^�ԍ����X�g��ݒ肷��.
			�ݒ肳�ꂽ�l������GetDInputBuffer(),GetOutputBuffer()�̖߂�l�����肷��.
			@param i_lpBatchDataNoList	�ݒ肷��f�[�^�ԍ����X�g. [GetBatchSize()�̖߂�l]�̗v�f�����K�v */
		Gravisbell::ErrorCode SetBatchDataNoList(const U32 i_lpBatchDataNoList[])
		{
			this->lpBatchDataNoList = i_lpBatchDataNoList;

			U32 outputBufferCount = this->GetOutputBufferCount();

			// �f�[�^���o�͗p�o�b�t�@�ɃR�s�[
			for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
			{
				if(this->lpBatchDataNoList[batchNum] > this->lpBufferList.size())
					return Gravisbell::ErrorCode::ERROR_CODE_COMMON_OUT_OF_ARRAYRANGE;

				U32 dataNo = this->lpBatchDataNoList[batchNum];

				// Device����Async���g���ƒx���Ȃ����̂Œʏ�R�s�[
				cudaMemcpy(
					thrust::raw_pointer_cast(&this->lpOutputBuffer[batchNum * outputBufferCount]),
					this->lpBufferList[dataNo],
					sizeof(F32) * outputBufferCount,
					cudaMemcpyDeviceToDevice);
			}

			return Gravisbell::ErrorCode::ERROR_CODE_NONE;
		}
	};

	/** ���͐M���f�[�^���C���[���쐬����.GUID�͎������蓖��.CPU����
		@param bufferSize	�o�b�t�@�̃T�C�Y.��F32�^�z��̗v�f��.
		@return	���͐M���f�[�^���C���[�̃A�h���X */
	extern IODataLayer_API Gravisbell::Layer::IOData::IIODataLayer* CreateIODataLayerGPU_device(Gravisbell::IODataStruct ioDataStruct)
	{
		boost::uuids::uuid uuid = boost::uuids::random_generator()();

		return CreateIODataLayerGPUwithGUID_device(uuid.data, ioDataStruct);
	}
	/** ���͐M���f�[�^���C���[���쐬����.CPU����
		@param guid			���C���[��GUID.
		@param bufferSize	�o�b�t�@�̃T�C�Y.��F32�^�z��̗v�f��.
		@return	���͐M���f�[�^���C���[�̃A�h���X */
	extern IODataLayer_API Gravisbell::Layer::IOData::IIODataLayer* CreateIODataLayerGPUwithGUID_device(Gravisbell::GUID guid, Gravisbell::IODataStruct ioDataStruct)
	{
		return new Gravisbell::Layer::IOData::IODataLayerGPU_device(guid, ioDataStruct);
	}

}	// IOData
}	// Layer
}	// Gravisbell
