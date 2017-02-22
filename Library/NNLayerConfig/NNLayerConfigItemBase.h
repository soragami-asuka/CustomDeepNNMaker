//==================================
// 設定項目
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
		/** コンストラクタ */
		NNLayerConfigItemBase(const wchar_t i_szID[], const wchar_t i_szName[], const wchar_t i_szText[])
			:	id		(i_szID)
			,	name	(i_szName)
			,	text	(i_szText)
		{
		}
		/** コピーコンストラクタ */
		NNLayerConfigItemBase(const NNLayerConfigItemBase& item)
			:	id		(item.id)
			,	name	(item.name)
			,	text	(item.text)
		{
		}

		/** デストラクタ */
		virtual ~NNLayerConfigItemBase()
		{
		}

		/** 一致演算 */
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
		/** 不一致演算 */
		virtual bool operator!=(const NNLayerConfigItemBase& item)const
		{
			return !NNLayerConfigItemBase::operator==(item);
		}

	public:
		/** 項目IDを取得する.
			@param o_szIDBuf	IDを格納するバッファ. CONFIGITEM_NAME_MAXの文字数が必要 */
		ELayerErrorCode GetConfigID(wchar_t o_szIDBuf[])const
		{
			if(this->id.size() >= CONFIGITEM_ID_MAX)
				return LAYER_ERROR_COMMON_OUT_OF_ARRAYRANGE;

			wcscpy(o_szIDBuf, this->id.c_str());

			return LAYER_ERROR_NONE;
		}
		/** 項目名を取得する.
			@param o_szNameBuf	名前を格納するバッファ. CONFIGITEM_NAME_MAXの文字数が必要 */
		ELayerErrorCode GetConfigName(wchar_t o_szNameBuf[])const
		{
			if(this->name.size() >= CONFIGITEM_NAME_MAX)
				return LAYER_ERROR_COMMON_OUT_OF_ARRAYRANGE;

			wcscpy(o_szNameBuf, this->name.c_str());

			return LAYER_ERROR_NONE;
		}
		/** 項目の説明テキストを取得する.
			@param o_szBuf	説明文を格納するバッファ.CONFIGITEM_TEXT_MAXの文字数が必要. */
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