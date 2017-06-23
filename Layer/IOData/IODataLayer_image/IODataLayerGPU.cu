//===================================
// ���o�̓f�[�^���Ǘ�����N���X
// GPU����
// host�������m�ی^
//===================================
#include "stdafx.h"
#include "IODataLayerGPU_base.cuh"


#include<vector>
#include<list>
#include<algorithm>

// UUID�֘A�p
#include<boost/uuid/uuid_generators.hpp>

#define BLOCK_SIZE	(16)

using namespace Gravisbell;

namespace
{
	/** �x�N�g���̗v�f���m�̊|���Z. */
	__global__ void cuda_func_ConvertImage2Binaryr(const U08* i_lpInputBuffer, F32* o_lpOutputBuffer, U32 i_width, U32 i_height, U32 i_ch, U32 i_bachNum)
	{
		const U32 inputNum = blockIdx.x * BLOCK_SIZE + threadIdx.x;
		if(inputNum >= i_width*i_height)	// ���򂷂邪������warp�����Ȃ̂ŁA�������x�ɉe���͂Ȃ��͂�...
			return;
		
		const U32 batchPos = i_bachNum * i_width * i_height * i_ch;

		for(U32 ch=0; ch<i_ch; ch++)
		{
			U32 inputPos  = batchPos +  inputNum * i_ch + ch;
			U32 outputPos = batchPos +  ch * i_width * i_height + inputNum;

			o_lpOutputBuffer[outputPos] = i_lpInputBuffer[inputPos] / 0xFF;
		}
	}
}


namespace Gravisbell {
namespace Layer {
namespace IOData {

	class IODataLayerGPU : public IODataLayerGPU_base
	{
	private:
		std::vector<U08*> lpBufferList;


	public:
		/** �R���X�g���N�^ */
		IODataLayerGPU(Gravisbell::GUID guid, U32 i_dataCount, Gravisbell::IODataStruct ioDataStruct)
			:	IODataLayerGPU_base	(guid, ioDataStruct)
		{
			this->lpBufferList.resize(i_dataCount);
		}
		/** �f�X�g���N�^ */
		virtual ~IODataLayerGPU()
		{
			for(U32 i=0; i<lpBufferList.size(); i++)
			{
				if(lpBufferList[i] != NULL)
					delete lpBufferList[i];
			}
		}



		//==============================
		// �f�[�^�Ǘ��n
		//==============================
	public:
		/** �f�[�^��ǉ�����.
			@param	lpData	�f�[�^��g�̔z��. GetBufferSize()�̖߂�l�̗v�f�����K�v.
			@return	�ǉ����ꂽ�ۂ̃f�[�^�Ǘ��ԍ�. ���s�����ꍇ�͕��̒l. */
		Gravisbell::ErrorCode SetData(U32 i_dataNum, const BYTE lpData[], U32 i_lineLength)
		{
			if(lpData == NULL)
				return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;
			if(i_dataNum >= this->lpBufferList.size())
				return ErrorCode::ERROR_CODE_COMMON_OUT_OF_ARRAYRANGE;

			// �o�b�t�@�m��
			U08* lpBuffer = new U08[this->GetBufferCount()];
			if(lpBuffer == NULL)
				return ErrorCode::ERROR_CODE_COMMON_ALLOCATION_MEMORY;

			// �R�s�[
			for(U32 y=0; y<this->ioDataStruct.y; y++)
			{
				memcpy(&lpBuffer[y*this->ioDataStruct.x*this->ioDataStruct.ch], &lpData[y * i_lineLength], this->ioDataStruct.x*this->ioDataStruct.ch);
			}

			// ���X�g�ɒǉ�
			if(lpBufferList[i_dataNum])
				delete lpBufferList[i_dataNum];
			lpBufferList[i_dataNum] = lpBuffer;

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
		Gravisbell::ErrorCode GetDataByNum(U32 num, BYTE o_lpBufferList[])const
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

		/** �o�b�`�����f�[�^�ԍ����X�g��ݒ肷��.
			�ݒ肳�ꂽ�l������GetDInputBuffer(),GetOutputBuffer()�̖߂�l�����肷��.
			@param i_lpBatchDataNoList	�ݒ肷��f�[�^�ԍ����X�g. [GetBatchSize()�̖߂�l]�̗v�f�����K�v */
		Gravisbell::ErrorCode SetBatchDataNoList(const U32 i_lpBatchDataNoList[])
		{
			this->lpBatchDataNoList = i_lpBatchDataNoList;

			U32 outputBufferCount = this->GetOutputBufferCount();

			// �f�[�^���v�Z�p�ꎞ�o�b�t�@�ɃR�s�[
			for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
			{
				if(this->lpBatchDataNoList[batchNum] > this->lpBufferList.size())
				{
					cudaThreadSynchronize();
					return Gravisbell::ErrorCode::ERROR_CODE_COMMON_OUT_OF_ARRAYRANGE;
				}

				U32 dataNo = this->lpBatchDataNoList[batchNum];

				cudaMemcpyAsync(
					thrust::raw_pointer_cast(&this->lpTmpImageBuffer[batchNum * outputBufferCount]),
					this->lpBufferList[dataNo],
					sizeof(U32) * outputBufferCount,
					cudaMemcpyHostToDevice);
			}
			cudaThreadSynchronize();

			// �v�Z�p�ꎞ�o�b�t�@�Ɋi�[�����f�[�^��BYTE > F32, y,x,ch > ch,y,x�ϊ�����
			for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
			{
				U32 bufferCount = this->ioDataStruct.x * this->ioDataStruct.ch;
				dim3 grid((bufferCount +(BLOCK_SIZE - 1))/BLOCK_SIZE , 1, 1);
				dim3 block(BLOCK_SIZE, 1, 1);

				cuda_func_ConvertImage2Binaryr<<<grid, block>>>(
					thrust::raw_pointer_cast(&this->lpTmpImageBuffer[0]),
					thrust::raw_pointer_cast(&this->lpOutputBuffer[0]),
					ioDataStruct.x, ioDataStruct.y, ioDataStruct.ch,
					batchNum);
			}

			return Gravisbell::ErrorCode::ERROR_CODE_NONE;
		}
	};

	/** ���͐M���f�[�^���C���[���쐬����.GUID�͎������蓖��.CPU����
		@param bufferSize	�o�b�t�@�̃T�C�Y.��F32�^�z��̗v�f��.
		@return	���͐M���f�[�^���C���[�̃A�h���X */
	extern IODataLayer_image_API Gravisbell::Layer::IOData::IIODataLayer_image* CreateIODataLayerGPU(Gravisbell::U32 i_dataCount, Gravisbell::U32 i_width, Gravisbell::U32 i_height, Gravisbell::U32 i_ch)
	{
		boost::uuids::uuid uuid = boost::uuids::random_generator()();

		return CreateIODataLayerGPUwithGUID(uuid.data, i_dataCount, i_width, i_height, i_ch);
	}
	/** ���͐M���f�[�^���C���[���쐬����.CPU����
		@param guid			���C���[��GUID.
		@param bufferSize	�o�b�t�@�̃T�C�Y.��F32�^�z��̗v�f��.
		@return	���͐M���f�[�^���C���[�̃A�h���X */
	extern IODataLayer_image_API Gravisbell::Layer::IOData::IIODataLayer_image* CreateIODataLayerGPUwithGUID(Gravisbell::GUID guid, Gravisbell::U32 i_dataCount, Gravisbell::U32 i_width, Gravisbell::U32 i_height, Gravisbell::U32 i_ch)
	{
		return new Gravisbell::Layer::IOData::IODataLayerGPU(guid, i_dataCount, IODataStruct(i_ch, i_width, i_height, 1));
	}

}	// IOData
}	// Layer
}	// Gravisbell
