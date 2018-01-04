// TemporaryMemoryManager.cpp : DLL �A�v���P�[�V�����p�ɃG�N�X�|�[�g�����֐����`���܂��B
//

#include "stdafx.h"

#include"Library/Common/TemporaryMemoryManager.h"

#include<map>
#include<list>

#include<thrust/device_vector.h>


namespace Gravisbell {
namespace Common {

	template<class BufferType>
	class TemporaryMemoryManager : public ITemporaryMemoryManager
	{
	private:
		struct BufferInfo
		{
			bool onReserved;						/**< �g�p�\�񂪂���Ă��邩 */
			GUID guid;								/**< �\�񂵂Ă��郌�C���[��GUID */
			BufferType lpBuffer;	/**< �o�b�t�@�{�� */

			BufferInfo()
				:	onReserved	(false)
				,	guid		()
				,	lpBuffer()
			{
			}
			BufferInfo(BufferInfo& info)
				:	onReserved	(info.onReserved)
				,	guid		(info.guid)
				,	lpBuffer	(info.lpBuffer)
			{
			}
			BufferInfo(bool i_onReserved, GUID i_guid, U32 i_bufferSize)
				:	onReserved	(i_onReserved)
				,	guid		(i_guid)
				,	lpBuffer	(i_bufferSize)
			{
			}
		};

		std::map<GUID, std::map<std::wstring, U32>>	lpBufferSize;				/**< �o�b�t�@�̃T�C�Y�ꗗ */
		std::map<std::wstring, BufferType>	lpShareBuffer;		/**< ���L�o�b�t�@�{�� */
		std::map<std::wstring, std::list<BufferInfo>>		lpReserveBuffer;	/**< �\��o�b�t�@�{�� */


	public:
		/** �R���X�g���N�^ */
		TemporaryMemoryManager()
			:	ITemporaryMemoryManager()
		{
		}

		/** �f�X�g���N�^ */
		virtual ~TemporaryMemoryManager()
		{
		}

	public:
		/** �o�b�t�@�T�C�Y��o�^����.
			@param	i_layerGUID		���C���[��GUID.
			@param	i_szCode		�g�p���@���`����ID.
			@param	i_bufferSize	�o�b�t�@�̃T�C�Y. �o�C�g�P��. */
		ErrorCode SetBufferSize(GUID i_layerGUID, const wchar_t i_szCode[], U32 i_bufferSize)
		{
			this->lpBufferSize[i_layerGUID][(std::wstring)i_szCode] = i_bufferSize;

			return ErrorCode::ERROR_CODE_NONE;
		}

		/** �o�b�t�@�T�C�Y���擾����.
			@param	i_layerGUID		���C���[��GUID.
			@param	i_szCode		�g�p���@���`����ID. */
		U32 GetBufferSize(GUID i_layerGUID, const wchar_t i_szCode[])const
		{
			auto it_guid = this->lpBufferSize.find(i_layerGUID);
			if(it_guid == this->lpBufferSize.end())
				return 0;

			auto it_code = it_guid->second.find(i_szCode);
			if(it_code == it_guid->second.end())
				return 0;

			return it_code->second;
		}

		/** �o�b�t�@���擾���� */
		BYTE* GetBuffer(GUID i_layerGUID, const wchar_t i_szCode[])
		{
			U32 bufferSize = lpBufferSize[i_layerGUID][i_szCode];
			thrust::device_vector<BYTE>& buffer = this->lpShareBuffer[i_szCode];

			if(buffer.size() < bufferSize)
				buffer.resize(bufferSize);

			return thrust::raw_pointer_cast(&buffer[0]);
		}

		/** �o�b�t�@��\�񂵂Ď擾���� */
		BYTE* ReserveBuffer(GUID i_layerGUID, const wchar_t i_szCode[])
		{
			// �o�b�t�@�T�C�Y���擾
			U32 bufferSize = lpBufferSize[i_layerGUID][i_szCode];

			// ����R�[�h�̋󂫃o�b�t�@������
			auto& bufferList = this->lpReserveBuffer[i_szCode];
			auto it = bufferList.begin();
			while(it != bufferList.end())
			{
				if(!it->onReserved)
					break;
				if(it->guid == i_layerGUID)
					return thrust::raw_pointer_cast(&it->lpBuffer[0]);	// �\��ς݂œ���GUID�̏ꍇ�I��

				it++;
			}

			if(it == bufferList.end())
			{
				// �󂫂��Ȃ��ꍇ�͋����I�ɒǉ�
				it = bufferList.insert(bufferList.end(), BufferInfo(true, i_layerGUID, bufferSize));
			}
			else
			{
				// �o�b�t�@�̃T�C�Y���m�F���āA�K�v�Ȃ�g��
				if(it->lpBuffer.size() < bufferSize)
					it->lpBuffer.resize(bufferSize);
			}

			it->onReserved = true;
			it->guid = i_layerGUID;

			return thrust::raw_pointer_cast(&it->lpBuffer[0]);
		}
		/** �\��ς݃o�b�t�@���J������ */
		void RestoreBuffer(GUID i_layerGUID, const wchar_t i_szCode[])
		{
			// ����R�[�h�̗\��ς݃o�b�t�@������
			auto& bufferList = this->lpReserveBuffer[i_szCode];
			auto it = bufferList.begin();
			while(it != bufferList.end())
			{
				if(it->guid == i_layerGUID)
				{
					it->onReserved = false;
					return;
				}

				it++;
			}
		}
	};


	TemporaryMemoryManager_API ITemporaryMemoryManager* CreateTemporaryMemoryManagerGPU()
	{
		return new TemporaryMemoryManager<thrust::device_vector<BYTE>>();
	}
	TemporaryMemoryManager_API ITemporaryMemoryManager* CreateTemporaryMemoryManagerCPU()
	{
		return new TemporaryMemoryManager<thrust::host_vector<BYTE>>();
	}

}	// Common
}	// Gravisbell
