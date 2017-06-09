//===================================
// ���o�̓f�[�^���Ǘ�����N���X
// GPU����
// device�������m�ی^
//===================================


#include "stdafx.h"
#include "IODataSequentialLayerGPU_base.cuh"


#include<vector>
#include<list>
#include<algorithm>

// UUID�֘A�p
#include<boost/uuid/uuid_generators.hpp>


namespace Gravisbell {
namespace Layer {
namespace IOData {

	class IODataSequentialLayerGPU_device : public IODataSequentialLayerGPU_base
	{
	private:

	public:
		/** �R���X�g���N�^ */
		IODataSequentialLayerGPU_device(Gravisbell::GUID guid, Gravisbell::IODataStruct ioDataStruct)
			:	IODataSequentialLayerGPU_base	(guid, ioDataStruct)
		{
		}
		/** �f�X�g���N�^ */
		virtual ~IODataSequentialLayerGPU_device()
		{
		}


		//==============================
		// �f�[�^�Ǘ��n
		//==============================
	public:
		/** �f�[�^��ǉ�����.
			@param	lpData	�f�[�^��g�̔z��. GetBufferSize()�̖߂�l�̗v�f�����K�v.
			@return	�ǉ����ꂽ�ۂ̃f�[�^�Ǘ��ԍ�. ���s�����ꍇ�͕��̒l. */
		Gravisbell::ErrorCode SetData(U32 i_dataNo, const F32 lpData[])
		{
			if(lpData == NULL)
				return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;
			if(i_dataNo > this->batchSize)
				return ErrorCode::ERROR_CODE_COMMON_OUT_OF_ARRAYRANGE;

			// �R�s�[
			cudaMemcpy(
				thrust::raw_pointer_cast(&this->lpOutputBuffer[i_dataNo * this->GetOutputBufferCount()]),
				lpData,
				sizeof(F32) * this->GetOutputBufferCount(),
				cudaMemcpyHostToDevice);

			return Gravisbell::ErrorCode::ERROR_CODE_NONE;
		}
		/** �f�[�^��ǉ�����.
			@param	lpData	�f�[�^��g�̔z��. [GetBufferSize()�̖߂�l]�̗v�f�����K�v. 0�`255�̒l. �����I�ɂ�0.0�`1.0�ɕϊ������. */
		ErrorCode SetData(U32 i_dataNo, const BYTE lpData[])
		{
			if(lpData == NULL)
				return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;
			if(i_dataNo > this->batchSize)
				return ErrorCode::ERROR_CODE_COMMON_OUT_OF_ARRAYRANGE;

			// �R�s�[
			//cudaMemcpy(
			//	thrust::raw_pointer_cast(&this->lpOutputBuffer[i_dataNo * this->GetOutputBufferCount()]),
			//	lpData,
			//	sizeof(F32) * this->GetOutputBufferCount(),
			//	cudaMemcpyHostToDevice);

			return Gravisbell::ErrorCode::ERROR_CODE_NONE;
		}

		/** �f�[�^�����擾���� */
		U32 GetDataCount()const
		{
			return (U32)this->batchSize;
		}
		/** �f�[�^��ԍ��w��Ŏ擾����.
			@param num		�擾����ԍ�
			@param o_lpBufferList �f�[�^�̊i�[��z��. GetBufferSize()�̖߂�l�̗v�f�����K�v.
			@return ���������ꍇ0 */
		Gravisbell::ErrorCode GetDataByNum(U32 num, F32 o_lpBufferList[])const
		{
			if(num >= this->batchSize)
				return ErrorCode::ERROR_CODE_COMMON_OUT_OF_ARRAYRANGE;

			if(o_lpBufferList == NULL)
				return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

			cudaMemcpy(
				o_lpBufferList,
				thrust::raw_pointer_cast(&this->lpOutputBuffer[num*this->GetOutputBufferCount()]),
				sizeof(F32)*this->GetOutputBufferCount(),
				cudaMemcpyDeviceToHost);

			return Gravisbell::ErrorCode::ERROR_CODE_NONE;
		}
	};

	/** ���͐M���f�[�^���C���[���쐬����.GUID�͎������蓖��.CPU����
		@param bufferSize	�o�b�t�@�̃T�C�Y.��F32�^�z��̗v�f��.
		@return	���͐M���f�[�^���C���[�̃A�h���X */
	extern IODataSequentialLayer_API Gravisbell::Layer::IOData::IIODataSequentialLayer* CreateIODataSequentialLayerGPU_device(Gravisbell::IODataStruct ioDataStruct)
	{
		boost::uuids::uuid uuid = boost::uuids::random_generator()();

		return CreateIODataSequentialLayerGPUwithGUID_device(uuid.data, ioDataStruct);
	}
	/** ���͐M���f�[�^���C���[���쐬����.CPU����
		@param guid			���C���[��GUID.
		@param bufferSize	�o�b�t�@�̃T�C�Y.��F32�^�z��̗v�f��.
		@return	���͐M���f�[�^���C���[�̃A�h���X */
	extern IODataSequentialLayer_API Gravisbell::Layer::IOData::IIODataSequentialLayer* CreateIODataSequentialLayerGPUwithGUID_device(Gravisbell::GUID guid, Gravisbell::IODataStruct ioDataStruct)
	{
		return new Gravisbell::Layer::IOData::IODataSequentialLayerGPU_device(guid, ioDataStruct);
	}

}	// IOData
}	// Layer
}	// Gravisbell
