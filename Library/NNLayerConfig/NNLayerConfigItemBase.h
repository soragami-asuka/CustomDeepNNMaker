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
		std::wstring id;
		std::wstring name;
		std::wstring text;

	public:
		/** �R���X�g���N�^ */
		NNLayerConfigItemBase(const wchar_t i_szID[], const wchar_t i_szName[], const wchar_t i_szText[])
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
			@param o_szIDBuf	ID���i�[����o�b�t�@. CONFIGITEM_NAME_MAX�̕��������K�v */
		ELayerErrorCode GetConfigID(wchar_t o_szIDBuf[])const
		{
			if(this->id.size() >= CONFIGITEM_ID_MAX)
				return LAYER_ERROR_COMMON_OUT_OF_ARRAYRANGE;

			wcscpy(o_szIDBuf, this->id.c_str());

			return LAYER_ERROR_NONE;
		}
		/** ���ږ����擾����.
			@param o_szNameBuf	���O���i�[����o�b�t�@. CONFIGITEM_NAME_MAX�̕��������K�v */
		ELayerErrorCode GetConfigName(wchar_t o_szNameBuf[])const
		{
			if(this->name.size() >= CONFIGITEM_NAME_MAX)
				return LAYER_ERROR_COMMON_OUT_OF_ARRAYRANGE;

			wcscpy(o_szNameBuf, this->name.c_str());

			return LAYER_ERROR_NONE;
		}
		/** ���ڂ̐����e�L�X�g���擾����.
			@param o_szBuf	���������i�[����o�b�t�@.CONFIGITEM_TEXT_MAX�̕��������K�v. */
		ELayerErrorCode GetConfigText(wchar_t o_szBuf[])const
		{
			if(this->text.size() >= CONFIGITEM_TEXT_MAX)
				return LAYER_ERROR_COMMON_OUT_OF_ARRAYRANGE;

			wcscpy(o_szBuf, this->text.c_str());

			return LAYER_ERROR_NONE;
		}
	};
}


#endif // __NN_LAYER_CONFIG_ITEM_BASE_H__