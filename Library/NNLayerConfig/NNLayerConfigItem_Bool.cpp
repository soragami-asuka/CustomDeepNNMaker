//==================================
// �ݒ荀��(�_���^)
//==================================
#include "stdafx.h"

#include"NNLayerConfig.h"
#include"NNLayerConfigItemBase.h"

#include<string>
#include<vector>

namespace CustomDeepNNLibrary
{
	class NNLayerConfigItem_Bool : virtual public NNLayerConfigItemBase<INNLayerConfigItem_Bool>
	{
	private:
		bool defaultValue;

		bool value;

	public:
		/** �R���X�g���N�^ */
		NNLayerConfigItem_Bool(const wchar_t i_szID[], const wchar_t i_szName[], const wchar_t i_szText[], bool defaultValue)
			: NNLayerConfigItemBase(i_szID, i_szName, i_szText)
			, defaultValue(defaultValue)
			, value	(defaultValue)
		{
		}
		/** �R�s�[�R���X�g���N�^ */
		NNLayerConfigItem_Bool(const NNLayerConfigItem_Bool& item)
			: NNLayerConfigItemBase(item)
			, defaultValue	(item.defaultValue)
			, value			(item.value)
		{
		}
		/** �f�X�g���N�^ */
		virtual ~NNLayerConfigItem_Bool(){}
		
		/** ��v���Z */
		bool operator==(const INNLayerConfigItemBase& item)const
		{
			// ��ʂ̊m�F
			if(this->GetItemType() != item.GetItemType())
				return false;

			// �A�C�e����ϊ�
			const NNLayerConfigItem_Bool* pItem = dynamic_cast<const NNLayerConfigItem_Bool*>(&item);

			if(NNLayerConfigItemBase::operator!=(*pItem))
				return false;

			if(this->defaultValue != pItem->defaultValue)
				return false;

			if(this->value != pItem->value)
				return false;

			return true;
		}
		/** �s��v���Z */
		bool operator!=(const INNLayerConfigItemBase& item)const
		{
			return !(*this == item);
		}

		/** ���g�̕������쐬���� */
		INNLayerConfigItemBase* Clone()const
		{
			return new NNLayerConfigItem_Bool(*this);
		}

	public:
		/** �ݒ荀�ڎ�ʂ��擾���� */
		NNLayerConfigItemType GetItemType()const
		{
			return CONFIGITEM_TYPE_BOOL;
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
		ELayerErrorCode SetValue(bool value)
		{
			this->value = value;

			return LAYER_ERROR_NONE;
		}

	public:
		/** �f�t�H���g�̐ݒ�l���擾���� */
		bool GetDefault()const
		{
			return this->defaultValue;
		}


	public:
		/** �ۑ��ɕK�v�ȃo�C�g�����擾���� */
		unsigned int GetUseBufferByteCount()const
		{
			unsigned int byteCount = 0;

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

			unsigned int bufferPos = 0;

			// �l
			this->value = *(bool*)&i_lpBuffer[bufferPos];
			bufferPos += sizeof(this->value);


			return bufferPos;
		}
		/** �o�b�t�@�ɏ�������.
			@param o_lpBuffer	�������ݐ�o�b�t�@�̐擪�A�h���X. GetUseBufferByteCount�̖߂�l�̃o�C�g�����K�v
			@return ���������ꍇ0 */
		int WriteToBuffer(BYTE* o_lpBuffer)const
		{
			unsigned int bufferPos = 0;

			// �l
			*(bool*)&o_lpBuffer[bufferPos] = this->value;
			bufferPos += sizeof(bool);

			return bufferPos;
		}
	};
	
	/** �ݒ荀��(����)���쐬���� */
	extern "C" NNLAYERCONFIG_API INNLayerConfigItem_Bool* CreateLayerCofigItem_Bool(const wchar_t i_szID[], const wchar_t i_szName[], const wchar_t i_szText[], bool defaultValue)
	{
		return new NNLayerConfigItem_Bool(i_szID, i_szName, i_szText, defaultValue);
	}
}