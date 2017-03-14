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

	class Item_Float : public ItemBase<IItem_Float>
	{
	private:
		F32 minValue;
		F32 maxValue;
		F32 defaultValue;

		F32 value;

	public:
		/** �R���X�g���N�^ */
		Item_Float(const wchar_t i_szID[], const wchar_t i_szName[], const wchar_t i_szText[], F32 minValue, F32 maxValue, F32 defaultValue)
			: ItemBase(i_szID, i_szName, i_szText)
			, minValue(minValue)
			, maxValue(maxValue)
			, defaultValue(defaultValue)
			, value	(defaultValue)
		{
		}
		/** �R�s�[�R���X�g���N�^ */
		Item_Float(const Item_Float& item)
			: ItemBase(item)
			, minValue(item.minValue)
			, maxValue(item.maxValue)
			, defaultValue(item.defaultValue)
			, value	(item.value)
		{
		}
		/** �f�X�g���N�^ */
		virtual ~Item_Float(){}
		
		/** ��v���Z */
		bool operator==(const IItemBase& item)const
		{
			// ��ʂ̊m�F
			if(this->GetItemType() != item.GetItemType())
				return false;

			// �A�C�e����ϊ�
			const Item_Float* pItem = dynamic_cast<const Item_Float*>(&item);
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
			return new Item_Float(*this);
		}

	public:
		/** �ݒ荀�ڎ�ʂ��擾���� */
		ItemType GetItemType()const
		{
			return ITEMTYPE_FLOAT;
		}

	public:
		/** �l���擾���� */
		F32 GetValue()const
		{
			return this->value;
		}
		/** �l��ݒ肷��
			@param value	�ݒ肷��l
			@return ���������ꍇ0 */
		ErrorCode SetValue(F32 value)
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
		F32 GetMin()const
		{
			return this->minValue;
		}
		/** �ݒ�\�ő�l���擾���� */
		F32 GetMax()const
		{
			return this->maxValue;
		}

		/** �f�t�H���g�̐ݒ�l���擾���� */
		F32 GetDefault()const
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
		int ReadFromBuffer(const BYTE* i_lpBuffer, int i_bufferSize)
		{
			if(i_bufferSize < (int)this->GetUseBufferByteCount())
				return -1;

			U32 bufferPos = 0;

			// �l
			F32 value = *(F32*)&i_lpBuffer[bufferPos];
			bufferPos += sizeof(this->value);

			this->SetValue(value);

			return bufferPos;
		}
		/** �o�b�t�@�ɏ�������.
			@param o_lpBuffer	�������ݐ�o�b�t�@�̐擪�A�h���X. GetUseBufferByteCount�̖߂�l�̃o�C�g�����K�v
			@return ���������ꍇ0 */
		int WriteToBuffer(BYTE* o_lpBuffer)const
		{
			U32 bufferPos = 0;

			// �l
			*(F32*)&o_lpBuffer[bufferPos] = this->value;
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
			F32 value = this->GetValue();

			*(F32*)o_lpBuffer = value;

			return sizeof(F32);
		}
		/** �\���̂���ǂݍ���.
			@return	�g�p�����o�C�g��. */
		S32 ReadFromStruct(const BYTE* i_lpBuffer)
		{
			F32 value = *(const F32*)i_lpBuffer;

			this->SetValue(value);

			return sizeof(F32);
		}
	};
	
	/** �ݒ荀��(����)���쐬���� */
	extern "C" GRAVISBELL_SETTINGDATA_API IItem_Float* CreateItem_Float(const wchar_t i_szID[], const wchar_t i_szName[], const wchar_t i_szText[], F32 minValue, F32 maxValue, F32 defaultValue)
	{
		return new Item_Float(i_szID, i_szName, i_szText, minValue, maxValue, defaultValue);
	}

}	// Standard
}	// SettingData
}	// Gravisbell