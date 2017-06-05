//===================================
// ���o�̓f�[�^���Ǘ�����N���X
// GPU����
// host�������m�ی^
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

	class IODataSequentialLayerGPU_host : public IODataSequentialLayerGPU_base
	{
	private:
		std::vector<F32> lpBufferList;
		std::vector<F32*> lpBatchBufferListPointer;


	public:
		/** �R���X�g���N�^ */
		IODataSequentialLayerGPU_host(Gravisbell::GUID guid, Gravisbell::IODataStruct ioDataStruct)
			:	IODataSequentialLayerGPU_base	(guid, ioDataStruct)
		{
		}
		/** �f�X�g���N�^ */
		virtual ~IODataSequentialLayerGPU_host()
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
			if(i_dataNo >= this->lpBatchBufferListPointer.size())
				return ErrorCode::ERROR_CODE_COMMON_OUT_OF_ARRAYRANGE;

			// �R�s�[
			memcpy(lpBatchBufferListPointer[i_dataNo], lpData, sizeof(F32)*this->GetBufferCount());

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

			for(U32 i=0; i<this->GetBufferCount(); i++)
			{
				o_lpBufferList[i] = this->lpBatchBufferListPointer[num][i];
			}

			return Gravisbell::ErrorCode::ERROR_CODE_NONE;
		}

	};

	/** ���͐M���f�[�^���C���[���쐬����.GUID�͎������蓖��.CPU����
		@param bufferSize	�o�b�t�@�̃T�C�Y.��F32�^�z��̗v�f��.
		@return	���͐M���f�[�^���C���[�̃A�h���X */
	extern IODataSequentialLayer_API Gravisbell::Layer::IOData::IIODataSequentialLayer* CreateIODataSequentialLayerGPU_host(Gravisbell::IODataStruct ioDataStruct)
	{
		boost::uuids::uuid uuid = boost::uuids::random_generator()();

		return CreateIODataSequentialLayerGPUwithGUID_host(uuid.data, ioDataStruct);
	}
	/** ���͐M���f�[�^���C���[���쐬����.CPU����
		@param guid			���C���[��GUID.
		@param bufferSize	�o�b�t�@�̃T�C�Y.��F32�^�z��̗v�f��.
		@return	���͐M���f�[�^���C���[�̃A�h���X */
	extern IODataSequentialLayer_API Gravisbell::Layer::IOData::IIODataSequentialLayer* CreateIODataSequentialLayerGPUwithGUID_host(Gravisbell::GUID guid, Gravisbell::IODataStruct ioDataStruct)
	{
		return new Gravisbell::Layer::IOData::IODataSequentialLayerGPU_host(guid, ioDataStruct);
	}

}	// IOData
}	// Layer
}	// Gravisbell
