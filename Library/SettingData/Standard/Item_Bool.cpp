//==================================
// 設定項目(論理型)
//==================================
#include "stdafx.h"

#include"Library/SettingData/Standard.h"
#include"ItemBase.h"

#include<string>
#include<vector>

namespace Gravisbell {
namespace SettingData {
namespace Standard {

	class LayerConfigItem_Bool : virtual public ItemBase<IItem_Bool>
	{
	private:
		bool defaultValue;

		bool value;

	public:
		/** コンストラクタ */
		LayerConfigItem_Bool(const wchar_t i_szID[], const wchar_t i_szName[], const wchar_t i_szText[], bool defaultValue)
			: ItemBase(i_szID, i_szName, i_szText)
			, defaultValue(defaultValue)
			, value	(defaultValue)
		{
		}
		/** コピーコンストラクタ */
		LayerConfigItem_Bool(const LayerConfigItem_Bool& item)
			: ItemBase(item)
			, defaultValue	(item.defaultValue)
			, value			(item.value)
		{
		}
		/** デストラクタ */
		virtual ~LayerConfigItem_Bool(){}
		
		/** 一致演算 */
		bool operator==(const IItemBase& item)const
		{
			// 種別の確認
			if(this->GetItemType() != item.GetItemType())
				return false;

			// アイテムを変換
			const LayerConfigItem_Bool* pItem = dynamic_cast<const LayerConfigItem_Bool*>(&item);

			if(ItemBase::operator!=(*pItem))
				return false;

			if(this->defaultValue != pItem->defaultValue)
				return false;

			if(this->value != pItem->value)
				return false;

			return true;
		}
		/** 不一致演算 */
		bool operator!=(const IItemBase& item)const
		{
			return !(*this == item);
		}

		/** 自身の複製を作成する */
		IItemBase* Clone()const
		{
			return new LayerConfigItem_Bool(*this);
		}

	public:
		/** 設定項目種別を取得する */
		ItemType GetItemType()const
		{
			return ITEMTYPE_BOOL;
		}

	public:
		/** 値を取得する */
		bool GetValue()const
		{
			return this->value;
		}
		/** 値を設定する
			@param value	設定する値
			@return 成功した場合0 */
		ErrorCode SetValue(bool value)
		{
			this->value = value;

			return ERROR_CODE_NONE;
		}

	public:
		/** デフォルトの設定値を取得する */
		bool GetDefault()const
		{
			return this->defaultValue;
		}


	public:
		//================================
		// ファイル保存関連.
		// 文字列本体や列挙値のIDなど構造体には保存されない細かい情報を取り扱う.
		//================================

		/** 保存に必要なバイト数を取得する */
		U64 GetUseBufferByteCount()const
		{
			U32 byteCount = 0;

			byteCount += sizeof(this->value);			// 値

			return byteCount;
		}

		/** バッファから読み込む.
			@param i_lpBuffer	読み込みバッファの先頭アドレス.
			@param i_bufferSize	読み込み可能バッファのサイズ.
			@return	実際に読み取ったバッファサイズ. 失敗した場合は負の値 */
		S64 ReadFromBuffer(const BYTE* i_lpBuffer, S64 i_bufferSize)
		{
			if(i_bufferSize < (S64)this->GetUseBufferByteCount())
				return -1;

			U32 bufferPos = 0;

			// 値
			this->value = *(bool*)&i_lpBuffer[bufferPos];
			bufferPos += sizeof(this->value);


			return bufferPos;
		}
		/** バッファに書き込む.
			@param o_lpBuffer	書き込み先バッファの先頭アドレス. GetUseBufferByteCountの戻り値のバイト数が必要
			@return 成功した場合書き込んだバッファサイズ.失敗した場合は負の値 */
		S64 WriteToBuffer(BYTE* o_lpBuffer)const
		{
			U32 bufferPos = 0;

			// 値
			*(bool*)&o_lpBuffer[bufferPos] = this->value;
			bufferPos += sizeof(bool);

			return bufferPos;
		}

	public:
		//================================
		// 構造体を利用したデータの取り扱い.
		// 情報量が少ない代わりにアクセス速度が速い
		//================================

		/** 構造体に書き込む.
			@return	使用したバイト数. */
		S32 WriteToStruct(BYTE* o_lpBuffer)const
		{
			*(bool*)o_lpBuffer = this->value;

			return sizeof(bool);
		}
		/** 構造体から読み込む.
			@return	使用したバイト数. */
		S32 ReadFromStruct(const BYTE* i_lpBuffer)
		{
			this->value = *(const bool*)i_lpBuffer;

			return sizeof(bool);
		}
	};
	
	/** 設定項目(実数)を作成する */
	extern "C" GRAVISBELL_SETTINGDATA_API IItem_Bool* CreateItem_Bool(const wchar_t i_szID[], const wchar_t i_szName[], const wchar_t i_szText[], bool defaultValue)
	{
		return new LayerConfigItem_Bool(i_szID, i_szName, i_szText, defaultValue);
	}

}	// Standard
}	// SettingData
}	// Gravisbell