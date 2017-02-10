//==================================
// �ݒ荀��(������)
//==================================
#include "stdafx.h"

#include"NNLayerConfig.h"

#include<string>
#include<vector>

namespace CustomDeepNNLibrary
{
	class NNLayerConfigItem_String : public INNLayerConfigItem_String
	{
	private:
		std::string name;
		std::string id;

		std::string defaultValue;
		std::string value;

	public:
		/** �R���X�g���N�^ */
		NNLayerConfigItem_String(const char szName[], const char defaultValue[])
			: INNLayerConfigItem_String()
			, name (szName)
			, defaultValue(defaultValue)
			, value	(defaultValue)
		{
		}
		/** �R�s�[�R���X�g���N�^ */
		NNLayerConfigItem_String(const NNLayerConfigItem_String& item)
			: INNLayerConfigItem_String()
			, name (item.name)
			, defaultValue(item.defaultValue)
			, value	(item.value)
		{
		}
		/** �f�X�g���N�^ */
		virtual ~NNLayerConfigItem_String(){}
		
		/** ��v���Z */
		bool operator==(const INNLayerConfigItemBase& item)const
		{
			// ��ʂ̊m�F
			if(this->GetItemType() != item.GetItemType())
				return false;

			// �A�C�e����ϊ�
			NNLayerConfigItem_String* pItem = (NNLayerConfigItem_String*)&item;

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
			return new NNLayerConfigItem_String(*this);
		}
	public:
		/** �ݒ荀�ڎ�ʂ��擾���� */
		NNLayerConfigItemType GetItemType()const
		{
			return CONFIGITEM_TYPE_STRING;
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
		/** ������̒������擾���� */
		virtual unsigned int GetLength()const
		{
			return this->name.size();
		}
		/** �l���擾����.
			@param o_szBuf	�i�[��o�b�t�@
			@return ���������ꍇ0 */
		virtual int GetValue(char o_szBuf[])const
		{
			memcpy(o_szBuf, this->value.c_str(), this->value.size() + 1);

			return 0;
		}
		/** �l��ݒ肷��
			@param i_szBuf	�ݒ肷��l
			@return ���������ꍇ0 */
		virtual ELayerErrorCode SetValue(const char i_szBuf[])
		{
			this->value = i_szBuf;

			return LAYER_ERROR_NONE;
		}
		
	public:
		/** �f�t�H���g�̐ݒ�l���擾����
			@param o_szBuf	�i�[��o�b�t�@
			@return ���������ꍇ0 */
		int GetDefault(char o_szBuf[])const
		{
			memcpy(o_szBuf, this->defaultValue.c_str(), this->defaultValue.size() + 1);

			return 0;
		}
		
	public:
		/** �ۑ��ɕK�v�ȃo�C�g�����擾���� */
		unsigned int GetUseBufferByteCount()const
		{
			unsigned int byteCount = 0;

			byteCount += sizeof(unsigned int);		// �l�̃o�b�t�@�T�C�Y
			byteCount += this->value.size();		// �l

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
			{
				// �o�b�t�@�T�C�Y
				unsigned int bufferSize = *(unsigned int*)&i_lpBuffer[bufferPos];
				bufferPos += sizeof(unsigned int);

				// �l
				std::vector<char> szBuf(bufferSize+1, NULL);
				for(unsigned int i=0; i<bufferSize; i++)
				{
					szBuf[i] = i_lpBuffer[bufferPos++];
				}
				this->value = &szBuf[0];
			}


			return bufferPos;
		}
		/** �o�b�t�@�ɏ�������.
			@param o_lpBuffer	�������ݐ�o�b�t�@�̐擪�A�h���X. GetUseBufferByteCount�̖߂�l�̃o�C�g�����K�v
			@return ���������ꍇ0 */
		int WriteToBuffer(BYTE* o_lpBuffer)const
		{
			unsigned int bufferPos = 0;
			
			// �l
			{
				// �o�b�t�@�T�C�Y
				unsigned int bufferSize = this->value.size();;
				*(unsigned int*)&o_lpBuffer[bufferPos] = bufferSize;
				bufferPos += sizeof(unsigned int);

				// �l
				memcpy(&o_lpBuffer[bufferPos], this->value.c_str(), bufferSize);
				bufferPos += bufferSize;
			}


			return bufferPos;
		}
	};
	
	/** �ݒ荀��(����)���쐬���� */
	extern "C" NNLAYERCONFIG_API INNLayerConfigItem_String* CreateLayerCofigItem_String(const char szName[], const char szDefaultValue[])
	{
		return new NNLayerConfigItem_String(szName, szDefaultValue);
	}
}