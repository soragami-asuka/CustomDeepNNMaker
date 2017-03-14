//==================================
// �ݒ荀��(����)
//==================================
#include "stdafx.h"

#include"SettingData.h"
#include"ItemBase.h"

#include<string>
#include<vector>

namespace Gravisbell {
namespace SettingData {
namespace Standard {

	class Item_Int : public ItemBase<IItem_Int>
	{
	private:
		S32 minValue;
		S32 maxValue;
		S32 defaultValue;

		S32 value;

	public:
		/** �R���X�g���N�^ */
		Item_Int(const wchar_t i_szID[], const wchar_t i_szName[], const wchar_t i_szText[], S32 minValue, S32 maxValue, S32 defaultValue)
			: ItemBase(i_szID, i_szName, i_szText)
			, minValue(minValue)
			, maxValue(maxValue)
			, defaultValue(defaultValue)
			, value	(defaultValue)
		{
		}
		/** �R�s�[�R���X�g���N�^ */
		Item_Int(const Item_Int& item)
			: ItemBase(item)
			, minValue(item.minValue)
			, maxValue(item.maxValue)
			, defaultValue(item.defaultValue)
			, value	(item.value)
		{
		}
		/** �f�X�g���N�^ */
		virtual ~Item_Int(){}
		
		/** ��v���Z */
		bool operator==(const IItemBase& item)const
		{
			// ��ʂ̊m�F
			if(this->GetItemType() != item.GetItemType())
				return false;

			// �A�C�e����ϊ�
			const Item_Int* pItem = dynamic_cast<const Item_Int*>(&item);
			if(pItem == NULL)
				return false;

			// �x�[�X��r
			if(ItemBase::operator!=(*pItem))
				return false;

			if(this->minValue != pItem->minValue)
				return false;
			if(this->maxValue != pItem->maxValue)
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
			return new Item_Int(*this);
		}

	public:
		/** �ݒ荀�ڎ�ʂ��擾���� */
		ItemType GetItemType()const
		{
			return ITEMTYPE_INT;
		}

	public:
		/** �l���擾���� */
		S32 GetValue()const
		{
			return this->value;
		}
		/** �l��ݒ肷��
			@param value	�ݒ肷��l
			@return ���������ꍇ0 */
		ErrorCode SetValue(S32 value)
		{
			if(value < this->minValue)
				return ERROR_CODE_COMMON_OUT_OF_VALUERANGE;
			if(value > this->maxValue)
				return ERROR_CODE_COMMON_OUT_OF_VALUERANGE;

			this->value = value;

			return ERROR_CODE_NONE;
		}

	public:
		/** �ݒ�\�ŏ��l���擾���� */
		S32 GetMin()const
		{
			return this->minValue;
		}
		/** �ݒ�\�ő�l���擾���� */
		S32 GetMax()const
		{
			return this->maxValue;
		}

		/** �f�t�H���g�̐ݒ�l���擾���� */
		S32 GetDefault()const
		{
			return this->defaultValue;
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

			byteCount += sizeof(this->value);			// �l

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
			S32 value = *(S32*)&i_lpBuffer[bufferPos];
			bufferPos += sizeof(this->value);

			this->SetValue(value);

			return bufferPos;
		}
		/** �o�b�t�@�ɏ�������.
			@param o_lpBuffer	�������ݐ�o�b�t�@�̐擪�A�h���X. GetUseBufferByteCount�̖߂�l�̃o�C�g�����K�v
			@return ���������ꍇ0 */
		S32 WriteToBuffer(BYTE* o_lpBuffer)const
		{
			U32 bufferPos = 0;

			// �l
			*(S32*)&o_lpBuffer[bufferPos] = this->value;
			bufferPos += sizeof(this->value);

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

	/** �ݒ荀��(����)���쐬���� */
	extern "C" GRAVISBELL_SETTINGDATA_API IItem_Int* CreateItem_Int(const wchar_t i_szID[], const wchar_t i_szName[], const wchar_t i_szText[], S32 minValue, S32 maxValue, S32 defaultValue)
	{
		return new Item_Int(i_szID, i_szName, i_szText, minValue, maxValue, defaultValue);
	}

}	// Standard
}	// SettingData
}	// Gravisbell