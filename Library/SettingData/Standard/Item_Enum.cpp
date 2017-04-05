//==================================
// �ݒ荀��(�񋓌^)
//==================================
#include "stdafx.h"

#include"SettingData.h"
#include"IItemEx_Enum.h"
#include"ItemBase.h"

#include<string>
#include<vector>

namespace Gravisbell {
namespace SettingData {
namespace Standard {

	class Item_Enum : public ItemBase<IItemEx_Enum>
	{
	private:
		struct EnumItem
		{
			std::wstring id;
			std::wstring name;
			std::wstring text;

			/** �R���X�g���N�^ */
			EnumItem()
				: id (L"")
				, name (L"")
				, text (L"")
			{
			}
			/** �R���X�g���N�^ */
			EnumItem(const std::wstring& id, const std::wstring& name, const std::wstring& text)
				: id		(id)
				, name		(name)
				, text		(text)
			{
			}
			/** �R�s�[�R���X�g���N�^ */
			EnumItem(const EnumItem& item)
				: id		(item.id)
				, name		(item.name)
				, text		(item.text)
			{
			}

			/** =���Z */
			const EnumItem& operator=(const EnumItem& item)
			{
				this->id = item.id;
				this->name = item.name;
				this->text = item.text;

				return *this;
			}


			bool operator==(const EnumItem& item)const
			{
				if(this->id != item.id)
					return false;
				if(this->name != item.name)
					return false;
				if(this->text != item.text)
					return false;
				return true;
			}
			bool operator!=(const EnumItem& item)const
			{
				return !(*this == item);
			}
		};

	private:
		std::vector<EnumItem> lpEnumItem;

		std::wstring value;
		std::wstring defaultValue;


	public:
		/** �R���X�g���N�^ */
		Item_Enum(const wchar_t i_szID[], const wchar_t i_szName[], const wchar_t i_szText[])
			: ItemBase(i_szID, i_szName, i_szText)
		{
		}
		/** �R�s�[�R���X�g���N�^ */
		Item_Enum(const Item_Enum& item)
			: ItemBase(item)
			, lpEnumItem (item.lpEnumItem)
			, value (item.value)
			, defaultValue (item.defaultValue)
		{
		}
		/** �f�X�g���N�^ */
		virtual ~Item_Enum(){}
		
		/** ��v���Z */
		bool operator==(const IItemBase& item)const
		{
			// ��ʂ̊m�F
			if(this->GetItemType() != item.GetItemType())
				return false;

			// �A�C�e����ϊ�
			const Item_Enum* pItem = dynamic_cast<const Item_Enum*>(&item);
			if(pItem == NULL)
				return false;

			if(ItemBase::operator!=(*pItem))
				return false;

			for(U32 itemNum=0; itemNum<this->lpEnumItem.size(); itemNum++)
			{
				if(this->lpEnumItem[itemNum] != pItem->lpEnumItem[itemNum])
					return false;
			}
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
			return new Item_Enum(*this);
		}

	public:
		/** �ݒ荀�ڎ�ʂ��擾���� */
		ItemType GetItemType()const
		{
			return ITEMTYPE_ENUM;
		}

	public:
		/** �l���擾���� */
		S32 GetValue()const
		{
			return this->GetNumByID(this->value.c_str());
		}

		/** �l��ݒ肷��
			@param value	�ݒ肷��l
			@return ���������ꍇ0 */
		ErrorCode SetValue(S32 value)
		{
			if(value < 0)
				return ERROR_CODE_COMMON_OUT_OF_VALUERANGE;
			if(value >= (S32)this->lpEnumItem.size())
				return ERROR_CODE_COMMON_OUT_OF_VALUERANGE;

			this->value = this->lpEnumItem[value].id;

			return ERROR_CODE_NONE;
		}
		/** �l��ݒ肷��(������w��)
			@param i_szID	�ݒ肷��l(������w��)
			@return ���������ꍇ0 */
		ErrorCode SetValue(const wchar_t i_szEnumID[])
		{
			return this->SetValue(this->GetNumByID(i_szEnumID));
		}

	public:
		/** �񋓗v�f�����擾���� */
		U32 GetEnumCount()const
		{
			return this->lpEnumItem.size();
		}
		/** �񋓗v�fID��ԍ��w��Ŏ擾����.
			@param num		�v�f�ԍ�
			@param o_szBuf	�i�[��o�b�t�@
			@return ���������ꍇ0 */
		S32 GetEnumID(S32 num, wchar_t o_szBuf[])const
		{
			if(num < 0)
				return -1;
			if(num >= (S32)this->lpEnumItem.size())
				return -1;
			
			wcscpy(o_szBuf, this->lpEnumItem[num].id.c_str());

			return 0;
		}
		/** �񋓗v�fID��ԍ��w��Ŏ擾����.
			@param num		�v�f�ԍ�
			@param o_szBuf	�i�[��o�b�t�@
			@return ���������ꍇ0 */
		S32 GetEnumName(S32 num, wchar_t o_szBuf[])const
		{
			if(num < 0)
				return -1;
			if(num >= (S32)this->lpEnumItem.size())
				return -1;
			
			wcscpy(o_szBuf, this->lpEnumItem[num].name.c_str());

			return 0;
		}
		/** �񋓗v�f��������ԍ��w��Ŏ擾����.
			@param num		�v�f�ԍ�
			@param o_szBuf	�i�[��o�b�t�@
			@return ���������ꍇ0 */
		S32 GetEnumText(S32 num, wchar_t o_szBuf[])const
		{
			if(num < 0)
				return -1;
			if(num >= (S32)this->lpEnumItem.size())
				return -1;
			
			wcscpy(o_szBuf, this->lpEnumItem[num].text.c_str());

			return 0;
		}

		/** ID���w�肵�ė񋓗v�f�ԍ����擾����
			@param i_szBuf�@���ׂ��ID.
			@return ���������ꍇ0<=num<GetEnumCount�̒l. ���s�����ꍇ�͕��̒l���Ԃ�. */
		S32 GetNumByID(const wchar_t i_szEnumID[])const
		{
			std::wstring enumID = i_szEnumID;

			for(U32 itemNum=0; itemNum<this->lpEnumItem.size(); itemNum++)
			{
				if(this->lpEnumItem[itemNum].id == enumID)
					return (S32)itemNum;
			}

			return -1;
		}

		/** �f�t�H���g�̐ݒ�l���擾���� */
		S32 GetDefault()const
		{
			return this->GetNumByID(this->defaultValue.c_str());
		}

	public:
		/** �񋓒l��ǉ�����.
			����ID�����ɑ��݂���ꍇ���s����.
			@param szEnumID		�ǉ�����ID
			@param szEnumName	�ǉ����閼�O
			@param szComment	�ǉ�����R�����g��.
			@return ���������ꍇ�A�ǉ����ꂽ�A�C�e���̔ԍ����Ԃ�. ���s�����ꍇ�͕��̒l���Ԃ�. */
		S32 AddEnumItem(const wchar_t szEnumID[], const wchar_t szEnumName[], const wchar_t szComment[])
		{
			// ����ID�����݂��邩�m�F
			S32 sameID = this->GetNumByID(szEnumID);
			if(sameID >= 0)
				return -1;

			std::wstring id = szEnumID;
			if(id.size()+1 >= ID_BUFFER_MAX)
				return -1;

			// �ǉ�
			this->lpEnumItem.push_back(EnumItem(id, szEnumName, szComment));

			return this->lpEnumItem.size()-1;
		}

		/** �񋓒l���폜����.
			@param num	�폜����񋓔ԍ�
			@return ���������ꍇ0 */
		S32 EraseEnumItem(S32 num)
		{
			if(num < 0)
				return -1;
			if(num >= (S32)this->lpEnumItem.size())
				return -1;

			// iterator��i�߂�
			auto it = this->lpEnumItem.begin();
			for(S32 i=0; i<num; i++)
				it++;

			// �폜
			this->lpEnumItem.erase(it);

			return 0;
		}
		/** �񋓒l���폜����
			@param szEnumID �폜�����ID
			@return ���������ꍇ0 */
		S32 EraseEnumItem(const wchar_t szEnumID[])
		{
			return this->EraseEnumItem(this->GetNumByID(szEnumID));
		}

		/** �f�t�H���g�l��ݒ肷��.	�ԍ��w��.
			@param num �f�t�H���g�l�ɐݒ肷��ԍ�. 
			@return ���������ꍇ0 */
		S32 SetDefaultItem(S32 num)
		{
			wchar_t szEnumID[ID_BUFFER_MAX];

			// ID���擾����
			if(this->GetEnumID(num, szEnumID) < 0)
				return -1;

			this->defaultValue = szEnumID;

			return 0;
		}
		/** �f�t�H���g�l��ݒ肷��.	ID�w��. 
			@param szID �f�t�H���g�l�ɐݒ肷��ID. 
			@return ���������ꍇ0 */
		S32 SetDefaultItem(const wchar_t szEnumID[])
		{
			if(this->GetNumByID(szEnumID) < 0)
				return -1;

			this->defaultValue = szEnumID;

			return 0;
		}


	public:
		//================================
		// �t�@�C���ۑ��֘A.
		// ������{�̂�񋓒l��ID�ȂǍ\���̂ɂ͕ۑ�����Ȃ��ׂ���������舵��.
		//================================

		/** �ۑ��ɕK�v�ȃo�C�g�����擾���� */
		U32 GetUseBufferByteCount()const
		{
			U32 byteCount = 0;

			byteCount += sizeof(U32);		// �l�̃o�b�t�@�T�C�Y
			byteCount += this->value.size();		// �l

			return byteCount;
		}

		/** �o�b�t�@����ǂݍ���.
			@param i_lpBuffer	�ǂݍ��݃o�b�t�@�̐擪�A�h���X.
			@param i_bufferSize	�ǂݍ��݉\�o�b�t�@�̃T�C�Y.
			@return	���ۂɓǂݎ�����o�b�t�@�T�C�Y. ���s�����ꍇ�͕��̒l */
		S32 ReadFromBuffer(const BYTE* i_lpBuffer, S32 i_bufferSize)
		{
			if(i_bufferSize < (S32)this->GetUseBufferByteCount())
				return -1;

			U32 bufferPos = 0;

			// �l
			{
				// �o�b�t�@�T�C�Y
				U32 bufferSize = *(U32*)&i_lpBuffer[bufferPos];
				bufferPos += sizeof(U32);

				// �l
				std::vector<wchar_t> szBuf(bufferSize+1, NULL);
				for(U32 i=0; i<bufferSize; i++)
				{
					szBuf[i] = i_lpBuffer[bufferPos++];
				}
				std::wstring value = &szBuf[0];


				this->SetValue(value.c_str());
			}


			return bufferPos;
		}
		/** �o�b�t�@�ɏ�������.
			@param o_lpBuffer	�������ݐ�o�b�t�@�̐擪�A�h���X. GetUseBufferByteCount�̖߂�l�̃o�C�g�����K�v
			@return ���������ꍇ�������񂾃o�b�t�@�T�C�Y.���s�����ꍇ�͕��̒l */
		S32 WriteToBuffer(BYTE* o_lpBuffer)const
		{
			U32 bufferPos = 0;
			
			// �l
			{
				// �o�b�t�@�T�C�Y
				U32 bufferSize = this->value.size();;
				*(U32*)&o_lpBuffer[bufferPos] = bufferSize;
				bufferPos += sizeof(U32);

				// �l
				memcpy(&o_lpBuffer[bufferPos], this->value.c_str(), bufferSize);
				bufferPos += bufferSize;
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
			S32 value = this->GetValue();

			*(S32*)o_lpBuffer = value;

			return sizeof(S32);
		}
		/** �\���̂���ǂݍ���.
			@return	�g�p�����o�C�g��. */
		S32 ReadFromStruct(const BYTE* i_lpBuffer)
		{
			S32 value = *(const S32*)i_lpBuffer;

			this->SetValue(value);

			return sizeof(S32);
		}
	};
	
	/** �ݒ荀��(�񋓒l)���쐬���� */
	extern "C" GRAVISBELL_SETTINGDATA_API IItemEx_Enum* CreateItem_Enum(const wchar_t i_szID[], const wchar_t i_szName[], const wchar_t i_szText[])
	{
		return new Item_Enum(i_szID, i_szName, i_szText);
	}

}	// Standard
}	// SettingData
}	// Gravisbell