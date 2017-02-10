//==================================
// �ݒ荀��(�_���^)
//==================================
#include "stdafx.h"

#include"NNLayerConfig.h"

#include<string>
#include<vector>

namespace CustomDeepNNLibrary
{
	class NNLayerConfigItem_Bool : public INNLayerConfigItem_Bool
	{
	private:
		std::string name;
		std::string id;

		bool defaultValue;

		bool value;

	public:
		/** �R���X�g���N�^ */
		NNLayerConfigItem_Bool(const char szName[], bool defaultValue)
			: INNLayerConfigItem_Bool()
			, name (szName)
			, defaultValue(defaultValue)
			, value	(defaultValue)
		{
		}
		/** �R�s�[�R���X�g���N�^ */
		NNLayerConfigItem_Bool(const NNLayerConfigItem_Bool& item)
			: INNLayerConfigItem_Bool()
			, name			(item.name)
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
			NNLayerConfigItem_Bool* pItem = (NNLayerConfigItem_Bool*)&item;

			if(this->name != pItem->name)
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
			return CONFIGITEM_TYPE_INT;
		}
		/** ���ږ����擾����.
			@param o_szNameBuf	���O���i�[����o�b�t�@. CONFIGITEM_NAME_MAX�̃o�C�g�����K�v */
		ELayerErrorCode GetConfigName(char o_szNameBuf[])const
		{
			if(this->name.size() >= CONFIGITEM_NAME_MAX)
				return LAYER_ERROR_COMMON_OUT_OF_ARRAYRANGE;

			memcpy(o_szNameBuf, this->name.c_str(), this->name.size() + 1);

			return LAYER_ERROR_NONE;
		}
		/** ����ID���擾����.
			@param o_szIDBuf	ID���i�[����o�b�t�@. CONFIGITEM_NAME_MAX�̃o�C�g�����K�v */
		ELayerErrorCode GetConfigID(char o_szIDBuf[])const
		{
			if(this->name.size() >= CONFIGITEM_NAME_MAX)
				return LAYER_ERROR_COMMON_OUT_OF_ARRAYRANGE;

			memcpy(o_szIDBuf, this->name.c_str(), this->name.size() + 1);

			return LAYER_ERROR_NONE;
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
	extern "C" NNLAYERCONFIG_API INNLayerConfigItem_Bool* CreateLayerCofigItem_Bool(const char szName[], bool defaultValue)
	{
		return new NNLayerConfigItem_Bool(szName, defaultValue);
	}
}