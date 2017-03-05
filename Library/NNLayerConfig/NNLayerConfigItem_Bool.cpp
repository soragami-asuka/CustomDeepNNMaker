//==================================
// �ݒ荀��(�_���^)
//==================================
#include "stdafx.h"

#include"LayerConfig.h"
#include"LayerConfigItemBase.h"

#include<string>
#include<vector>

namespace Gravisbell {
namespace NeuralNetwork {

	class LayerConfigItem_Bool : virtual public LayerConfigItemBase<ILayerConfigItem_Bool>
	{
	private:
		bool defaultValue;

		bool value;

	public:
		/** �R���X�g���N�^ */
		LayerConfigItem_Bool(const wchar_t i_szID[], const wchar_t i_szName[], const wchar_t i_szText[], bool defaultValue)
			: LayerConfigItemBase(i_szID, i_szName, i_szText)
			, defaultValue(defaultValue)
			, value	(defaultValue)
		{
		}
		/** �R�s�[�R���X�g���N�^ */
		LayerConfigItem_Bool(const LayerConfigItem_Bool& item)
			: LayerConfigItemBase(item)
			, defaultValue	(item.defaultValue)
			, value			(item.value)
		{
		}
		/** �f�X�g���N�^ */
		virtual ~LayerConfigItem_Bool(){}
		
		/** ��v���Z */
		bool operator==(const ILayerConfigItemBase& item)const
		{
			// ��ʂ̊m�F
			if(this->GetItemType() != item.GetItemType())
				return false;

			// �A�C�e����ϊ�
			const LayerConfigItem_Bool* pItem = dynamic_cast<const LayerConfigItem_Bool*>(&item);

			if(LayerConfigItemBase::operator!=(*pItem))
				return false;

			if(this->defaultValue != pItem->defaultValue)
				return false;

			if(this->value != pItem->value)
				return false;

			return true;
		}
		/** �s��v���Z */
		bool operator!=(const ILayerConfigItemBase& item)const
		{
			return !(*this == item);
		}

		/** ���g�̕������쐬���� */
		ILayerConfigItemBase* Clone()const
		{
			return new LayerConfigItem_Bool(*this);
		}

	public:
		/** �ݒ荀�ڎ�ʂ��擾���� */
		LayerConfigItemType GetItemType()const
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
			this->value = *(bool*)&i_lpBuffer[bufferPos];
			bufferPos += sizeof(this->value);


			return bufferPos;
		}
		/** �o�b�t�@�ɏ�������.
			@param o_lpBuffer	�������ݐ�o�b�t�@�̐擪�A�h���X. GetUseBufferByteCount�̖߂�l�̃o�C�g�����K�v
			@return ���������ꍇ0 */
		int WriteToBuffer(BYTE* o_lpBuffer)const
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
	extern "C" NNLAYERCONFIG_API ILayerConfigItem_Bool* CreateLayerCofigItem_Bool(const wchar_t i_szID[], const wchar_t i_szName[], const wchar_t i_szText[], bool defaultValue)
	{
		return new LayerConfigItem_Bool(i_szID, i_szName, i_szText, defaultValue);
	}

}	// NeuralNetwork
}	// Gravisbell