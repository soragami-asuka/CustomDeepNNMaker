//==================================
// �ݒ荀��(�_���^)
//==================================
#include "stdafx.h"

#include"Library/SettingData/Standard.h"
#include"ItemBase.h"

#include<string>
#include<vector>

namespace Gravisbell {
namespace SettingData {
namespace Standard {

	class LayerConfigItem_Bool : virtual public ItemBase<IItem_Bool>
	{
	private:
		bool defaultValue;

		bool value;

	public:
		/** �R���X�g���N�^ */
		LayerConfigItem_Bool(const wchar_t i_szID[], const wchar_t i_szName[], const wchar_t i_szText[], bool defaultValue)
			: ItemBase(i_szID, i_szName, i_szText)
			, defaultValue(defaultValue)
			, value	(defaultValue)
		{
		}
		/** �R�s�[�R���X�g���N�^ */
		LayerConfigItem_Bool(const LayerConfigItem_Bool& item)
			: ItemBase(item)
			, defaultValue	(item.defaultValue)
			, value			(item.value)
		{
		}
		/** �f�X�g���N�^ */
		virtual ~LayerConfigItem_Bool(){}
		
		/** ��v���Z */
		bool operator==(const IItemBase& item)const
		{
			// ��ʂ̊m�F
			if(this->GetItemType() != item.GetItemType())
				return false;

			// �A�C�e����ϊ�
			const LayerConfigItem_Bool* pItem = dynamic_cast<const LayerConfigItem_Bool*>(&item);

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
			return new LayerConfigItem_Bool(*this);
		}

	public:
		/** �ݒ荀�ڎ�ʂ��擾���� */
		ItemType GetItemType()const
		{
			return ITEMTYPE_BOOL;
		}

	public:
		/** �l���擾���� */
		bool GetValue()const
		{
			return this->value;
		}
		/** �l��ݒ肷��
			@param value	�ݒ肷��l
			@return ���������ꍇ0 */
		ErrorCode SetValue(bool value)
		{
			this->value = value;

			return ERROR_CODE_NONE;
		}

	public:
		/** �f�t�H���g�̐ݒ�l���擾���� */
		bool GetDefault()const
		{
			return this->defaultValue;
		}


	public:
		//================================
		// �t�@�C���ۑ��֘A.
		// ������{�̂�񋓒l��ID�ȂǍ\���̂ɂ͕ۑ�����Ȃ��ׂ���������舵��.
		//================================

		/** �ۑ��ɕK�v�ȃo�C�g�����擾���� */
		U64 GetUseBufferByteCount()const
		{
			U32 byteCount = 0;

			byteCount += sizeof(this->value);			// �l

			return byteCount;
		}

		/** �o�b�t�@����ǂݍ���.
			@param i_lpBuffer	�ǂݍ��݃o�b�t�@�̐擪�A�h���X.
			@param i_bufferSize	�ǂݍ��݉\�o�b�t�@�̃T�C�Y.
			@return	���ۂɓǂݎ�����o�b�t�@�T�C�Y. ���s�����ꍇ�͕��̒l */
		S64 ReadFromBuffer(const BYTE* i_lpBuffer, S64 i_bufferSize)
		{
			if(i_bufferSize < (S64)this->GetUseBufferByteCount())
				return -1;

			U32 bufferPos = 0;

			// �l
			this->value = *(bool*)&i_lpBuffer[bufferPos];
			bufferPos += sizeof(this->value);


			return bufferPos;
		}
		/** �o�b�t�@�ɏ�������.
			@param o_lpBuffer	�������ݐ�o�b�t�@�̐擪�A�h���X. GetUseBufferByteCount�̖߂�l�̃o�C�g�����K�v
			@return ���������ꍇ�������񂾃o�b�t�@�T�C�Y.���s�����ꍇ�͕��̒l */
		S64 WriteToBuffer(BYTE* o_lpBuffer)const
		{
			U32 bufferPos = 0;

			// �l
			*(bool*)&o_lpBuffer[bufferPos] = this->value;
			bufferPos += sizeof(bool);

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
			*(bool*)o_lpBuffer = this->value;

			return sizeof(bool);
		}
		/** �\���̂���ǂݍ���.
			@return	�g�p�����o�C�g��. */
		S32 ReadFromStruct(const BYTE* i_lpBuffer)
		{
			this->value = *(const bool*)i_lpBuffer;

			return sizeof(bool);
		}
	};
	
	/** �ݒ荀��(����)���쐬���� */
	extern "C" GRAVISBELL_SETTINGDATA_API IItem_Bool* CreateItem_Bool(const wchar_t i_szID[], const wchar_t i_szName[], const wchar_t i_szText[], bool defaultValue)
	{
		return new LayerConfigItem_Bool(i_szID, i_szName, i_szText, defaultValue);
	}

}	// Standard
}	// SettingData
}	// Gravisbell