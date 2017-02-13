//==================================
// �ݒ荀��
//==================================
#ifndef __NN_LAYER_CONFIG_ITEM_BASE_H__
#define __NN_LAYER_CONFIG_ITEM_BASE_H__


#include"NNLayerConfig.h"

#include<string>
#include<vector>


namespace CustomDeepNNLibrary
{
	template<class ItemType>
	class NNLayerConfigItemBase : virtual public ItemType
	{
	private:
		std::string id;
		std::string name;
		std::string text;

	public:
		/** �R���X�g���N�^ */
		NNLayerConfigItemBase(const char i_szID[], const char i_szName[], const char i_szText[])
			:	id		(i_szID)
			,	name	(i_szName)
			,	text	(i_szText)
		{
		}
		/** �R�s�[�R���X�g���N�^ */
		NNLayerConfigItemBase(const NNLayerConfigItemBase& item)
			:	id		(item.id)
			,	name	(item.name)
			,	text	(item.text)
		{
		}

		/** �f�X�g���N�^ */
		virtual ~NNLayerConfigItemBase()
		{
		}

		/** ��v���Z */
		virtual bool operator==(const NNLayerConfigItemBase& item)const
		{
			if(this->id != item.id)
				return false;
			if(this->name != item.name)
				return false;
			if(this->text != item.text)
				return false;

			return true;
		}
		/** �s��v���Z */
		virtual bool operator!=(const NNLayerConfigItemBase& item)const
		{
			return !NNLayerConfigItemBase::operator==(item);
		}

	public:
		/** ����ID���擾����.
			@param o_szIDBuf	ID���i�[����o�b�t�@. CONFIGITEM_NAME_MAX�̃o�C�g�����K�v */
		ELayerErrorCode GetConfigID(char o_szIDBuf[])const
		{
			if(this->name.size() >= CONFIGITEM_ID_MAX)
				return LAYER_ERROR_COMMON_OUT_OF_ARRAYRANGE;

			memcpy(o_szIDBuf, this->name.c_str(), this->name.size() + 1);

			return LAYER_ERROR_NONE;
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
		/** ���ڂ̐����e�L�X�g���擾����.
			@param o_szBuf	���������i�[����o�b�t�@.CONFIGITEM_TEXT_MAX�̃o�C�g�����K�v. */
		ELayerErrorCode GetConfigText(char o_szBuf[])const
		{
			if(this->text.size() >= CONFIGITEM_TEXT_MAX)
				return LAYER_ERROR_COMMON_OUT_OF_ARRAYRANGE;

			memcpy(o_szBuf, this->text.c_str(), this->text.size() + 1);

			return LAYER_ERROR_NONE;
		}
	};
}


#endif // __NN_LAYER_CONFIG_ITEM_BASE_H__