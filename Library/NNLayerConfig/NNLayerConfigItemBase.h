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
		std::string id;
		std::string name;
		std::string text;

	public:
		/** コンストラクタ */
		NNLayerConfigItemBase(const char i_szID[], const char i_szName[], const char i_szText[])
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
			@param o_szIDBuf	IDを格納するバッファ. CONFIGITEM_NAME_MAXのバイト数が必要 */
		ELayerErrorCode GetConfigID(char o_szIDBuf[])const
		{
			if(this->name.size() >= CONFIGITEM_ID_MAX)
				return LAYER_ERROR_COMMON_OUT_OF_ARRAYRANGE;

			memcpy(o_szIDBuf, this->name.c_str(), this->name.size() + 1);

			return LAYER_ERROR_NONE;
		}
		/** 項目名を取得する.
			@param o_szNameBuf	名前を格納するバッファ. CONFIGITEM_NAME_MAXのバイト数が必要 */
		ELayerErrorCode GetConfigName(char o_szNameBuf[])const
		{
			if(this->name.size() >= CONFIGITEM_NAME_MAX)
				return LAYER_ERROR_COMMON_OUT_OF_ARRAYRANGE;

			memcpy(o_szNameBuf, this->name.c_str(), this->name.size() + 1);

			return LAYER_ERROR_NONE;
		}
		/** 項目の説明テキストを取得する.
			@param o_szBuf	説明文を格納するバッファ.CONFIGITEM_TEXT_MAXのバイト数が必要. */
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