// LayerConfig.cpp : DLL �A�v���P�[�V�����p�ɃG�N�X�|�[�g�����֐����`���܂��B
//

#include "stdafx.h"
#include "LayerConfig.h"

#include<string>
#include<vector>

namespace Gravisbell {
namespace NeuralNetwork {

	class LayerConfig : public ILayerConfigEx
	{
	private:
		GUID layerCode;
		VersionCode versionCode;

		std::vector<ILayerConfigItemBase*> lpLayerConfigItem;

	public:
		/** �R���X�g���N�^ */
		LayerConfig(const GUID& layerCode, const VersionCode& versionCode)
			: ILayerConfigEx()
			, layerCode		(layerCode)
			, versionCode	(versionCode)
		{
		}
		/** �R�s�[�R���X�g���N�^ */
		LayerConfig(const LayerConfig& config)
			: layerCode	(config.layerCode)
			, versionCode (config.versionCode)
		{
			for(unsigned int itemNum=0; itemNum<config.lpLayerConfigItem.size(); itemNum++)
			{
				this->lpLayerConfigItem.push_back(config.lpLayerConfigItem[itemNum]->Clone());
			}
		}
		/** �f�X�g���N�^ */
		virtual ~LayerConfig()
		{
			for(unsigned int itemNum=0; itemNum<lpLayerConfigItem.size(); itemNum++)
			{
				if(lpLayerConfigItem[itemNum] != NULL)
					delete lpLayerConfigItem[itemNum];
			}
		}

		/** ��v���Z */
		bool operator==(const ILayerConfig& config)const
		{
			LayerConfig* pConfig = (LayerConfig*)&config;

			// ���C���[�R�[�h�̊m�F
			if(this->layerCode != pConfig->layerCode)
				return false;
			// �o�[�W�����R�[�h�̊m�F
			if(this->versionCode != pConfig->versionCode)
				return false;

			// �A�C�e�����̊m�F
			if(this->lpLayerConfigItem.size() != pConfig->lpLayerConfigItem.size())
				return false;

			// �e�A�C�e���̊m�F
			for(unsigned int itemNum=0; itemNum<this->lpLayerConfigItem.size(); itemNum++)
			{
				ILayerConfigItemBase* pItemL = this->lpLayerConfigItem[itemNum];
				ILayerConfigItemBase* pItemR = pConfig->lpLayerConfigItem[itemNum];

				// �ǂ��炩�Е���NULL�������ꍇ�͏I��
				if((pItemL == NULL) ^ (pItemR == NULL))
					return false;

				// ����NULL�������ꍇ�͈�v�Ɣ���
				if((pItemL == NULL) && (pItemR == NULL))
					continue;

				// �e�A�C�e�����m�F
				if(*pItemL != *pItemR)
					return false;
			}

			return true;
		}
		/** �s��v���Z */
		bool operator!=(const ILayerConfig& config)const
		{
			return !(*this == config);
		}

		/** ���g�̕������쐬���� */
		virtual ILayerConfig* Clone()const
		{
			return new LayerConfig(*this);
		}

	public:
		/** ���C���[���ʃR�[�h���擾����.
			@param o_layerCode	�i�[��o�b�t�@
			@return ���������ꍇ0 */
		ErrorCode GetLayerCode(GUID& o_guid)const
		{
			o_guid = this->layerCode;

			return ERROR_CODE_NONE;
		}

	public:
		/** �ݒ荀�ڐ����擾���� */
		unsigned int GetItemCount()const
		{
			return this->lpLayerConfigItem.size();
		}
		/** �ݒ荀�ڂ�ԍ��w��Ŏ擾���� */
		ILayerConfigItemBase* GetItemByNum(U32 i_num)
		{
			if(i_num >= this->lpLayerConfigItem.size())
				return NULL;
			return this->lpLayerConfigItem[i_num];
		}
		/** �ݒ荀�ڂ�ԍ��w��Ŏ擾���� */
		const ILayerConfigItemBase* GetItemByNum(U32 i_num)const
		{
			if(i_num >= this->lpLayerConfigItem.size())
				return NULL;
			return this->lpLayerConfigItem[i_num];
		}
		/** �ݒ荀�ڂ�ID�w��Ŏ擾���� */
		const ILayerConfigItemBase* GetItemByID(const wchar_t i_szIDBuf[])const
		{
			// ����ID������
			for(unsigned int i=0; i<this->lpLayerConfigItem.size(); i++)
			{
				wchar_t szID[CONFIGITEM_NAME_MAX];

				// �Ώۍ��ڂ�ID���擾
				if(this->lpLayerConfigItem[i]->GetConfigID(szID) != ErrorCode::ERROR_CODE_NONE)
					continue;

				// ��r
				if(std::wstring(szID) == i_szIDBuf)
					return this->lpLayerConfigItem[i];
			}
			return NULL;
		}
		
		/** �A�C�e����ǉ�����.
			�ǉ����ꂽ�A�C�e���͓�����delete�����. */
		int AddItem(ILayerConfigItemBase* pItem)
		{
			if(pItem == NULL)
				return -1;

			this->lpLayerConfigItem.push_back(pItem);

			return 0;
		}

	public:
		/** �ۑ��ɕK�v�ȃo�C�g�����擾���� */
		unsigned int GetUseBufferByteCount()const
		{
			unsigned int bufferSize = 0;

			// ���C���[�R�[�h
			bufferSize += sizeof(this->layerCode);

			// �o�[�W�����R�[�h
			bufferSize += sizeof(this->versionCode);

			// �A�C�e����
			bufferSize += sizeof(unsigned int);

			// �e�A�C�e��
			for(unsigned int itemNum=0; itemNum<this->lpLayerConfigItem.size(); itemNum++)
			{
				// �A�C�e�����
				bufferSize += sizeof(LayerConfigItemType);

				// �A�C�e��
				bufferSize += this->lpLayerConfigItem[itemNum]->GetUseBufferByteCount();
			}

			return bufferSize;
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

			// ���C���[�R�[�h
			GUID tmpLayerCode = *(GUID*)&i_lpBuffer[bufferPos];
			bufferPos += sizeof(GUID);

			// �`�F�b�N
			if(this->layerCode != tmpLayerCode)
				return -1;


			// �o�[�W�����R�[�h
			VersionCode tmpVersionCode = *(VersionCode*)&i_lpBuffer[bufferPos];
			bufferPos += sizeof(this->versionCode);

			// �`�F�b�N
			if(this->versionCode.major != tmpVersionCode.major)
				return -1;
			if(this->versionCode.minor != tmpVersionCode.minor)
				return -1;
			if(this->versionCode.revision != tmpVersionCode.revision)
				return -1;

			this->versionCode = tmpVersionCode;


			// �A�C�e����
			unsigned int itemCount = *(unsigned int*)&i_lpBuffer[bufferPos];
			bufferPos += sizeof(unsigned int);

			// �`�F�b�N
			if(this->lpLayerConfigItem.size() != itemCount)
				return -1;



			std::vector<ILayerConfigItemBase*> lpTmpItem;
			for(unsigned int itemNum=0; itemNum<itemCount; itemNum++)
			{
				// �A�C�e�����
				LayerConfigItemType itemType = *(LayerConfigItemType*)&i_lpBuffer[bufferPos];
				bufferPos += sizeof(LayerConfigItemType);

				if(this->lpLayerConfigItem[itemNum]->GetItemType() != itemType)
					return -1;

				// �ǂݍ���
				bufferPos += this->lpLayerConfigItem[itemNum]->ReadFromBuffer(&i_lpBuffer[bufferPos], i_bufferSize - bufferPos);
			}


			return bufferPos;
		}
		/** �o�b�t�@�ɏ�������.
			@param o_lpBuffer	�������ݐ�o�b�t�@�̐擪�A�h���X. GetUseBufferByteCount�̖߂�l�̃o�C�g�����K�v
			@return ���������ꍇ0 */
		int WriteToBuffer(BYTE* o_lpBuffer)const
		{
			unsigned int bufferPos = 0;
			
			// ���C���[�R�[�h
			*(GUID*)&o_lpBuffer[bufferPos] = this->layerCode;
			bufferPos += sizeof(GUID);

			// �o�[�W�����R�[�h
			*(VersionCode*)&o_lpBuffer[bufferPos] = this->versionCode;
			bufferPos += sizeof(VersionCode);

			// �A�C�e����
			*(unsigned int*)&o_lpBuffer[bufferPos] = this->lpLayerConfigItem.size();
			bufferPos += sizeof(unsigned int);

			// �e�A�C�e��
			for(unsigned int itemNum=0; itemNum<this->lpLayerConfigItem.size(); itemNum++)
			{
				// �A�C�e�����
				*(LayerConfigItemType*)&o_lpBuffer[bufferPos] = this->lpLayerConfigItem[itemNum]->GetItemType();
				bufferPos += sizeof(LayerConfigItemType);

				// �A�C�e��
				bufferPos += this->lpLayerConfigItem[itemNum]->WriteToBuffer(&o_lpBuffer[bufferPos]);
			}

			return bufferPos;
		}

	public:
		/** �\���̂Ƀf�[�^���i�[����.
			@param	o_lpBuffer	�\���̂̐擪�A�h���X. �\���̂�ConvertNNCofigToSource����o�͂��ꂽ�\�[�X���g�p����. */
		ErrorCode WriteToStruct(BYTE* o_lpBuffer)const
		{
			unsigned int bufferPos = 0;
			
			// �e�A�C�e��
			for(unsigned int itemNum=0; itemNum<this->lpLayerConfigItem.size(); itemNum++)
			{
				const ILayerConfigItemBase* pItemBase = this->GetItemByNum(itemNum);
				if(pItemBase == NULL)
					continue;

				bufferPos += pItemBase->WriteToStruct(&o_lpBuffer[bufferPos]);
			}

			return ErrorCode::ERROR_CODE_NONE;
		}
		/** �\���̂���f�[�^��ǂݍ���.
			@param	o_lpBuffer	�\���̂̐擪�A�h���X. �\���̂�ConvertNNCofigToSource����o�͂��ꂽ�\�[�X���g�p����. */
		ErrorCode ReadFromStruct(const BYTE* i_lpBuffer)
		{
			unsigned int bufferPos = 0;
			
			// �e�A�C�e��
			for(unsigned int itemNum=0; itemNum<this->lpLayerConfigItem.size(); itemNum++)
			{
				ILayerConfigItemBase* pItemBase = this->GetItemByNum(itemNum);
				if(pItemBase == NULL)
					continue;

				bufferPos += pItemBase->ReadFromStruct(&i_lpBuffer[bufferPos]);
			}

			return ErrorCode::ERROR_CODE_NONE;
		}
	};

	/** ��̃��C���[�ݒ�����쐬���� */
	extern "C" NNLAYERCONFIG_API ILayerConfigEx* CreateEmptyLayerConfig(const GUID& layerCode, const VersionCode& versionCode)
	{
		return new LayerConfig(layerCode, versionCode);
	}

}	// NeuralNetwork
}	// Gravisbell