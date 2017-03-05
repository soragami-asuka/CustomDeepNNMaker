//==================================
// �ݒ荀��
//==================================
#ifndef __GRAVISBELL_NN_LAYER_CONFIG_ITEM_BASE_H__
#define __GRAVISBELL_NN_LAYER_CONFIG_ITEM_BASE_H__


#include"LayerConfig.h"

#include<string>
#include<vector>


namespace Gravisbell {
namespace NeuralNetwork {

	template<class ItemType>
	class LayerConfigItemBase : virtual public ItemType
	{
	private:
		std::wstring id;
		std::wstring name;
		std::wstring text;

	public:
		/** �R���X�g���N�^ */
		LayerConfigItemBase(const wchar_t i_szID[], const wchar_t i_szName[], const wchar_t i_szText[])
			:	id		(i_szID)
			,	name	(i_szName)
			,	text	(i_szText)
		{
		}
		/** �R�s�[�R���X�g���N�^ */
		LayerConfigItemBase(const LayerConfigItemBase& item)
			:	id		(item.id)
			,	name	(item.name)
			,	text	(item.text)
		{
		}

		/** �f�X�g���N�^ */
		virtual ~LayerConfigItemBase()
		{
		}

		/** ��v���Z */
		virtual bool operator==(const LayerConfigItemBase& item)const
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
		virtual bool operator!=(const LayerConfigItemBase& item)const
		{
			return !LayerConfigItemBase::operator==(item);
		}

	public:
		/** ����ID���擾����.
			@param o_szIDBuf	ID���i�[����o�b�t�@. CONFIGITEM_NAME_MAX�̕��������K�v */
		ErrorCode GetConfigID(wchar_t o_szIDBuf[])const
		{
			if(this->id.size() >= CONFIGITEM_ID_MAX)
				return ERROR_CODE_COMMON_OUT_OF_ARRAYRANGE;

			wcscpy(o_szIDBuf, this->id.c_str());

			return ERROR_CODE_NONE;
		}
		/** ���ږ����擾����.
			@param o_szNameBuf	���O���i�[����o�b�t�@. CONFIGITEM_NAME_MAX�̕��������K�v */
		ErrorCode GetConfigName(wchar_t o_szNameBuf[])const
		{
			if(this->name.size() >= CONFIGITEM_NAME_MAX)
				return ERROR_CODE_COMMON_OUT_OF_ARRAYRANGE;

			wcscpy(o_szNameBuf, this->name.c_str());

			return ERROR_CODE_NONE;
		}
		/** ���ڂ̐����e�L�X�g���擾����.
			@param o_szBuf	���������i�[����o�b�t�@.CONFIGITEM_TEXT_MAX�̕��������K�v. */
		ErrorCode GetConfigText(wchar_t o_szBuf[])const
		{
			if(this->text.size() >= CONFIGITEM_TEXT_MAX)
				return ERROR_CODE_COMMON_OUT_OF_ARRAYRANGE;

			wcscpy(o_szBuf, this->text.c_str());

			return ERROR_CODE_NONE;
		}
	};

}	// NeuralNetwork
}	// Gravisbell


#endif // __NN_LAYER_CONFIG_ITEM_BASE_H__