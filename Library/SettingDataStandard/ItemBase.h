//==================================
// 設定項目
//==================================
#ifndef __GRAVISBELL_NN_LAYER_CONFIG_ITEM_BASE_H__
#define __GRAVISBELL_NN_LAYER_CONFIG_ITEM_BASE_H__

#include"SettingData/Standard/IData.h"


#include<string>
#include<vector>


namespace Gravisbell {
namespace SettingData {
namespace Standard {

	template<class ItemType>
	class ItemBase : virtual public ItemType
	{
	private:
		std::wstring id;
		std::wstring name;
		std::wstring text;

	public:
		/** コンストラクタ */
		ItemBase(const wchar_t i_szID[], const wchar_t i_szName[], const wchar_t i_szText[])
			:	id		(i_szID)
			,	name	(i_szName)
			,	text	(i_szText)
		{
		}
		/** コピーコンストラクタ */
		ItemBase(const ItemBase& item)
			:	id		(item.id)
			,	name	(item.name)
			,	text	(item.text)
		{
		}

		/** デストラクタ */
		virtual ~ItemBase()
		{
		}

		/** 一致演算 */
		virtual bool operator==(const ItemBase& item)const
		{
			if(this->id != item.id)
				return false;
			if(this->name != item.name)
				return false;
			if(this->text != item.text)
				return false;

			return true;
		}
		/** 不一致演算 */
		virtual bool operator!=(const ItemBase& item)const
		{
			return !ItemBase::operator==(item);
		}

	public:
		/** 項目IDを取得する.
			@param o_szIDBuf	IDを格納するバッファ. CONFIGITEM_NAME_MAXの文字数が必要 */
		ErrorCode GetConfigID(wchar_t o_szIDBuf[])const
		{
			if(this->id.size() >= ITEM_ID_MAX)
				return ERROR_CODE_COMMON_OUT_OF_ARRAYRANGE;

			wcscpy(o_szIDBuf, this->id.c_str());

			return ERROR_CODE_NONE;
		}
		/** 項目名を取得する.
			@param o_szNameBuf	名前を格納するバッファ. CONFIGITEM_NAME_MAXの文字数が必要 */
		ErrorCode GetConfigName(wchar_t o_szNameBuf[])const
		{
			if(this->name.size() >= ITEM_NAME_MAX)
				return ERROR_CODE_COMMON_OUT_OF_ARRAYRANGE;

			wcscpy(o_szNameBuf, this->name.c_str());

			return ERROR_CODE_NONE;
		}
		/** 項目の説明テキストを取得する.
			@param o_szBuf	説明文を格納するバッファ.CONFIGITEM_TEXT_MAXの文字数が必要. */
		ErrorCode GetConfigText(wchar_t o_szBuf[])const
		{
			if(this->text.size() >= ITEM_TEXT_MAX)
				return ERROR_CODE_COMMON_OUT_OF_ARRAYRANGE;

			wcscpy(o_szBuf, this->text.c_str());

			return ERROR_CODE_NONE;
		}
	};

}	// Standard
}	// SettingData
}	// Gravisbell


#endif // __NN_LAYER_CONFIG_ITEM_BASE_H__