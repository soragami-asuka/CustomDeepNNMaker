//=======================================
// ���C���[DLL�N���X
//=======================================
#ifndef __GRAVISBELL_I_TEMPORARY_MEMORY_MANAGER_H__
#define __GRAVISBELL_I_TEMPORARY_MEMORY_MANAGER_H__

#include"Guiddef.h"
#include"ErrorCode.h"

namespace Gravisbell {
namespace Common {

	class ITemporaryMemoryManager
	{
	public:
		/** �R���X�g���N�^ */
		ITemporaryMemoryManager(){}
		/** �f�X�g���N�^ */
		virtual ~ITemporaryMemoryManager(){}

	public:
		/** �o�b�t�@�T�C�Y��o�^����.
			@param	i_layerGUID		���C���[��GUID.
			@param	i_szCode		�g�p���@���`����ID.
			@param	i_bufferSize	�o�b�t�@�̃T�C�Y. �o�C�g�P��. */
		virtual ErrorCode SetBufferSize(GUID i_layerGUID, const wchar_t i_szCode[], U32 i_bufferSize) = 0;

		/** �o�b�t�@�T�C�Y���擾����.
			@param	i_layerGUID		���C���[��GUID.
			@param	i_szCode		�g�p���@���`����ID.
			@param	i_bufferSize	�o�b�t�@�̃T�C�Y. �o�C�g�P��. */
		virtual U32 GetBufferSize(GUID i_layerGUID, const wchar_t i_szCode[])const = 0;

		/** �o�b�t�@���擾���� */
//		virtual BYTE* GetBuffer(GUID i_layerGUID, const wchar_t i_szCode[]) = 0;

		/** �o�b�t�@��\�񂵂Ď擾���� */
		virtual BYTE* ReserveBuffer(GUID i_layerGUID, const wchar_t i_szCode[]) = 0;
		/** �\��ς݃o�b�t�@���J������ */
		virtual void RestoreBuffer(GUID i_layerGUID, const wchar_t i_szCode[]) = 0;
	};

}	// Layer
}	// Gravisbell

#endif