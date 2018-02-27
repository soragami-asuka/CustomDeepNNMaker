//==================================
// �ݒ荀��(������)
//==================================
#include "stdafx.h"

#include"Library/SettingData/Standard.h"
#include"ItemBase.h"

#include<string>
#include<vector>

namespace Gravisbell {
namespace SettingData {
namespace Standard {

	class Item_String : public ItemBase<IItem_String>
	{
	private:
		std::wstring defaultValue;
		std::wstring value;

	public:
		/** �R���X�g���N�^ */
		Item_String(const wchar_t i_szID[], const wchar_t i_szName[], const wchar_t i_szText[], const wchar_t defaultValue[])
			: ItemBase(i_szID, i_szName, i_szText)
			, defaultValue(defaultValue)
			, value	(defaultValue)
		{
		}
		/** �R�s�[�R���X�g���N�^ */
		Item_String(const Item_String& item)
			: ItemBase(item)
			, defaultValue(item.defaultValue)
			, value	(item.value)
		{
		}
		/** �f�X�g���N�^ */
		virtual ~Item_String(){}
		
		/** ��v���Z */
		bool operator==(const IItemBase& item)const
		{
			// ��ʂ̊m�F
			if(this->GetItemType() != item.GetItemType())
				return false;

			// �A�C�e����ϊ�
			const Item_String* pItem = dynamic_cast<const Item_String*>(&item);
			if(pItem == NULL)
				return false;

			// �x�[�X��r
			if(ItemBase::operator!=(*pItem))
				return false;

			if(this->defaultValue != pItem->defaultValue)
				return false;

			if(this->value != pItem->value)
				return false;

			return true;
		}
		/** �s��v���Z */
		bool operator!=(const IItemBase& item)const
		{
			return !(*this == item);
		}

		/** ���g�̕������쐬���� */
		IItemBase* Clone()const
		{
			return new Item_String(*this);
		}
	public:
		/** �ݒ荀�ڎ�ʂ��擾���� */
		ItemType GetItemType()const
		{
			return ITEMTYPE_STRING;
		}

	public:
		/** ������̒������擾���� */
		virtual unsigned int GetLength()const
		{
			return (U32)this->value.size();
		}
		/** �l���擾����.
			@param o_szBuf	�i�[��o�b�t�@
			@return ���������ꍇ0 */
		virtual int GetValue(wchar_t o_szBuf[])const
		{
			wcscpy(o_szBuf, this->value.c_str());

			return 0;
		}
		/** �l���擾����.
			@return ���������ꍇ������̐擪�A�h���X. */
		virtual const wchar_t* GetValue()const
		{
			return this->value.c_str();
		}
		/** �l��ݒ肷��
			@param i_szBuf	�ݒ肷��l
			@return ���������ꍇ0 */
		virtual ErrorCode SetValue(const wchar_t i_szBuf[])
		{
			this->value = i_szBuf;

			return ERROR_CODE_NONE;
		}
		
	public:
		/** �f�t�H���g�̐ݒ�l���擾����
			@param o_szBuf	�i�[��o�b�t�@
			@return ���������ꍇ0 */
		int GetDefault(wchar_t o_szBuf[])const
		{
			wcscpy(o_szBuf, this->defaultValue.c_str());

			return 0;
		}
		
	public:
		//================================
		// �t�@�C���ۑ��֘A.
		// ������{�̂�񋓒l��ID�ȂǍ\���̂ɂ͕ۑ�����Ȃ��ׂ���������舵��.
		//================================

		/** �ۑ��ɕK�v�ȃo�C�g�����擾���� */
		U64 GetUseBufferByteCount()const
		{
			unsigned int byteCount = 0;

			byteCount += sizeof(unsigned int);		// �l�̃o�b�t�@�T�C�Y
			byteCount += (U32)this->value.size() * sizeof(wchar_t);		// �l

			return byteCount;
		}

		/** �o�b�t�@����ǂݍ���.
			@param i_lpBuffer	�ǂݍ��݃o�b�t�@�̐擪�A�h���X.
			@param i_bufferSize	�ǂݍ��݉\�o�b�t�@�̃T�C�Y.
			@return	���ۂɓǂݎ�����o�b�t�@�T�C�Y. ���s�����ꍇ�͕��̒l */
		S64 ReadFromBuffer(const BYTE* i_lpBuffer, S64 i_bufferSize)
		{
			if(i_bufferSize < (int)this->GetUseBufferByteCount())
				return -1;

			unsigned int bufferPos = 0;

			// �l
			{
				// �o�b�t�@�T�C�Y
				unsigned int bufferSize = *(unsigned int*)&i_lpBuffer[bufferPos];
				bufferPos += sizeof(unsigned int);

				// �l
				std::vector<wchar_t> szBuf(bufferSize+1, NULL);
				for(unsigned int i=0; i<bufferSize; i++)
				{
					szBuf[i] = *(wchar_t*)&i_lpBuffer[bufferPos];

					bufferPos += sizeof(wchar_t);
				}
				this->value = &szBuf[0];
			}


			return bufferPos;
		}
		/** �o�b�t�@�ɏ�������.
			@param o_lpBuffer	�������ݐ�o�b�t�@�̐擪�A�h���X. GetUseBufferByteCount�̖߂�l�̃o�C�g�����K�v
			@return ���������ꍇ�������񂾃o�b�t�@�T�C�Y.���s�����ꍇ�͕��̒l */
		S64 WriteToBuffer(BYTE* o_lpBuffer)const
		{
			unsigned int bufferPos = 0;
			
			// �l
			{
				// �o�b�t�@�T�C�Y
				unsigned int bufferSize = (U32)this->value.size();
				*(unsigned int*)&o_lpBuffer[bufferPos] = bufferSize;
				bufferPos += sizeof(unsigned int);

				// �l
				memcpy(&o_lpBuffer[bufferPos], this->value.c_str(), bufferSize * sizeof(wchar_t));
				bufferPos += bufferSize * sizeof(wchar_t);
			}


			return bufferPos;
		}
		
	public:
		//================================
		// �\���̂𗘗p�����f�[�^�̎�舵��.
		// ���ʂ����Ȃ�����ɃA�N�Z�X���x������
		//================================

		/** �\���̂ɏ�������.
			@return	�g�p�����o�C�g��. */
		S32 WriteToStruct(BYTE* o_lpBuffer)const
		{
			const wchar_t* value = this->GetValue();

			*(const wchar_t**)o_lpBuffer = value;

			return sizeof(const wchar_t*);
		}
		/** �\���̂���ǂݍ���.
			@return	�g�p�����o�C�g��. */
		S32 ReadFromStruct(const BYTE* i_lpBuffer)
		{
			const wchar_t* value = *(const wchar_t**)i_lpBuffer;

			this->SetValue(value);

			return sizeof(const wchar_t*);
		}
	};
	
	/** �ݒ荀��(����)���쐬���� */
	extern "C" GRAVISBELL_SETTINGDATA_API IItem_String* CreateItem_String(const wchar_t i_szID[], const wchar_t i_szName[], const wchar_t i_szText[], const wchar_t szDefaultValue[])
	{
		return new Item_String(i_szID, i_szName, i_szText, szDefaultValue);
	}

}	// Standard
}	// SettingData
}	// Gravisbell