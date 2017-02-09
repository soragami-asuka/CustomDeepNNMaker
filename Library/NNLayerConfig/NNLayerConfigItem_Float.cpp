//==================================
// �ݒ荀��(����)
//==================================
#include "stdafx.h"

#include"NNLayerConfig.h"

#include<string>
#include<vector>

namespace CustomDeepNNLibrary
{
	class NNLayerConfigItem_Float : public INNLayerConfigItem_Float
	{
	private:
		std::string name;

		float minValue;
		float maxValue;
		float defaultValue;

		float value;

	public:
		/** �R���X�g���N�^ */
		NNLayerConfigItem_Float(const char szName[], float minValue, float maxValue, float defaultValue)
			: INNLayerConfigItem_Float()
			, name (szName)
			, minValue(minValue)
			, maxValue(maxValue)
			, defaultValue(defaultValue)
			, value	(defaultValue)
		{
		}
		/** �R�s�[�R���X�g���N�^ */
		NNLayerConfigItem_Float(const NNLayerConfigItem_Float& item)
			: INNLayerConfigItem_Float()
			, name (item.name)
			, minValue(item.minValue)
			, maxValue(item.maxValue)
			, defaultValue(item.defaultValue)
			, value	(item.value)
		{
		}
		/** �f�X�g���N�^ */
		virtual ~NNLayerConfigItem_Float(){}
		
		/** ��v���Z */
		bool operator==(const INNLayerConfigItemBase& item)const
		{
			// ��ʂ̊m�F
			if(this->GetItemType() != item.GetItemType())
				return false;

			// �A�C�e����ϊ�
			NNLayerConfigItem_Float* pItem = (NNLayerConfigItem_Float*)&item;

			if(this->name != pItem->name)
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
		bool operator!=(const INNLayerConfigItemBase& item)const
		{
			return !(*this == item);
		}

		/** ���g�̕������쐬���� */
		INNLayerConfigItemBase* Clone()const
		{
			return new NNLayerConfigItem_Float(*this);
		}

	public:
		/** �ݒ荀�ڎ�ʂ��擾���� */
		NNLayerConfigItemType GetItemType()const
		{
			return CONFIGITEM_TYPE_FLOAT;
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

	public:
		/** �l���擾���� */
		float GetValue()const
		{
			return this->value;
		}
		/** �l��ݒ肷��
			@param value	�ݒ肷��l
			@return ���������ꍇ0 */
		ELayerErrorCode SetValue(float value)
		{
			if(value < this->minValue)
				return LAYER_ERROR_COMMON_OUT_OF_VALUERANGE;
			if(value > this->maxValue)
				return LAYER_ERROR_COMMON_OUT_OF_VALUERANGE;

			this->value = value;

			return LAYER_ERROR_NONE;
		}

	public:
		/** �ݒ�\�ŏ��l���擾���� */
		float GetMin()const
		{
			return this->minValue;
		}
		/** �ݒ�\�ő�l���擾���� */
		float GetMax()const
		{
			return this->maxValue;
		}

		/** �f�t�H���g�̐ݒ�l���擾���� */
		float GetDefault()const
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
			float value = *(float*)&i_lpBuffer[bufferPos];
			bufferPos += sizeof(this->value);

			this->SetValue(value);

			return bufferPos;
		}
		/** �o�b�t�@�ɏ�������.
			@param o_lpBuffer	�������ݐ�o�b�t�@�̐擪�A�h���X. GetUseBufferByteCount�̖߂�l�̃o�C�g�����K�v
			@return ���������ꍇ0 */
		int WriteToBuffer(BYTE* o_lpBuffer)const
		{
			unsigned int bufferPos = 0;

			// �l
			*(float*)&o_lpBuffer[bufferPos] = this->value;
			bufferPos += sizeof(this->value);

			return bufferPos;
		}
	};
	
	/** �ݒ荀��(����)���쐬���� */
	extern "C" NNLAYERCONFIG_API INNLayerConfigItem_Float* CreateLayerCofigItem_Float(const char szName[], float minValue, float maxValue, float defaultValue)
	{
		return new NNLayerConfigItem_Float(szName, minValue, maxValue, defaultValue);
	}
}