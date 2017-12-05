// TemporaryMemoryManager.cpp : DLL �A�v���P�[�V�����p�ɃG�N�X�|�[�g�����֐����`���܂��B
//

#include "stdafx.h"

#include"Library/Common/TemporaryMemoryManager.h"

#include<map>
#include<vector>

namespace Gravisbell {
namespace Common {

	class TemporaryMemoryManager_CPU : public ITemporaryMemoryManager
	{
	private:
		std::map<GUID, std::map<std::wstring, U32>>	lpBufferSize;	/**< �o�b�t�@�̃T�C�Y�ꗗ */
		std::map<std::wstring, std::vector<BYTE>>	lpBuffer;		/**< �o�b�t�@�{�� */

	public:
		/** �R���X�g���N�^ */
		TemporaryMemoryManager_CPU()
			:	ITemporaryMemoryManager()
		{
		}

		/** �f�X�g���N�^ */
		virtual ~TemporaryMemoryManager_CPU()
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
		BYTE* GetBufer(GUID i_layerGUID, const wchar_t i_szCode[])
		{
			U32 bufferSize = lpBufferSize[i_layerGUID][i_szCode];
			std::vector<BYTE>& buffer = this->lpBuffer[i_szCode];

			if(buffer.size() < bufferSize)
				buffer.resize(bufferSize);

			return &buffer[0];
		}
	};


	TemporaryMemoryManager_API ITemporaryMemoryManager* CreateTemporaryMemoryManagerCPU()
	{
		return new TemporaryMemoryManager_CPU();
	}

}	// Common
}	// Gravisbell
