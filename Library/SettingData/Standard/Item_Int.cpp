//==================================
// 設定項目(整数)
//==================================
#include "stdafx.h"

#include"SettingData.h"
#include"ItemBase.h"

#include<string>
#include<vector>

namespace Gravisbell {
namespace SettingData {
namespace Standard {

	class Item_Int : public ItemBase<IItem_Int>
	{
	private:
		S32 minValue;
		S32 maxValue;
		S32 defaultValue;

		S32 value;

	public:
		/** コンストラクタ */
		Item_Int(const wchar_t i_szID[], const wchar_t i_szName[], const wchar_t i_szText[], S32 minValue, S32 maxValue, S32 defaultValue)
			: ItemBase(i_szID, i_szName, i_szText)
			, minValue(minValue)
			, maxValue(maxValue)
			, defaultValue(defaultValue)
			, value	(defaultValue)
		{
		}
		/** コピーコンストラクタ */
		Item_Int(const Item_Int& item)
			: ItemBase(item)
			, minValue(item.minValue)
			, maxValue(item.maxValue)
			, defaultValue(item.defaultValue)
			, value	(item.value)
		{
		}
		/** デストラクタ */
		virtual ~Item_Int(){}
		
		/** 一致演算 */
		bool operator==(const IItemBase& item)const
		{
			// 種別の確認
			if(this->GetItemType() != item.GetItemType())
				return false;

			// アイテムを変換
			const Item_Int* pItem = dynamic_cast<const Item_Int*>(&item);
			if(pItem == NULL)
				return false;

			// ベース比較
			if(ItemBase::operator!=(*pItem))
				return false;

			if(this->minValue != pItem->minValue)
				return false;
			if(this->maxValue != pItem->maxValue)
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
			return new Item_Int(*this);
		}

	public:
		/** 設定項目種別を取得する */
		ItemType GetItemType()const
		{
			return ITEMTYPE_INT;
		}

	public:
		/** 値を取得する */
		S32 GetValue()const
		{
			return this->value;
		}
		/** 値を設定する
			@param value	設定する値
			@return 成功した場合0 */
		ErrorCode SetValue(S32 value)
		{
			if(value < this->minValue)
				return ERROR_CODE_COMMON_OUT_OF_VALUERANGE;
			if(value > this->maxValue)
				return ERROR_CODE_COMMON_OUT_OF_VALUERANGE;

			this->value = value;

			return ERROR_CODE_NONE;
		}

	public:
		/** 設定可能最小値を取得する */
		S32 GetMin()const
		{
			return this->minValue;
		}
		/** 設定可能最大値を取得する */
		S32 GetMax()const
		{
			return this->maxValue;
		}

		/** デフォルトの設定値を取得する */
		S32 GetDefault()const
		{
			return this->defaultValue;
		}
		
	public:
		//================================
		// ファイル保存関連.
		// 文字列本体や列挙値のIDなど構造体には保存されない細かい情報を取り扱う.
		//================================

		/** 保存に必要なバイト数を取得する */
		U32 GetUseBufferByteCount()const
		{
			U32 byteCount = 0;

			byteCount += sizeof(this->value);			// 値

			return byteCount;
		}

		/** バッファから読み込む.
			@param i_lpBuffer	読み込みバッファの先頭アドレス.
			@param i_bufferSize	読み込み可能バッファのサイズ.
			@return	実際に読み取ったバッファサイズ. 失敗した場合は負の値 */
		S32 ReadFromBuffer(const BYTE* i_lpBuffer, S32 i_bufferSize)
		{
			if(i_bufferSize < (S32)this->GetUseBufferByteCount())
				return -1;

			U32 bufferPos = 0;

			// 値
			S32 value = *(S32*)&i_lpBuffer[bufferPos];
			bufferPos += sizeof(this->value);

			this->SetValue(value);

			return bufferPos;
		}
		/** バッファに書き込む.
			@param o_lpBuffer	書き込み先バッファの先頭アドレス. GetUseBufferByteCountの戻り値のバイト数が必要
			@return 成功した場合0 */
		S32 WriteToBuffer(BYTE* o_lpBuffer)const
		{
			U32 bufferPos = 0;

			// 値
			*(S32*)&o_lpBuffer[bufferPos] = this->value;
			bufferPos += sizeof(this->value);

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
			S32 value = this->GetValue();

			*(S32*)o_lpBuffer = value;

			return sizeof(S32);
		}
		/** 構造体から読み込む.
			@return	使用したバイト数. */
		S32 ReadFromStruct(const BYTE* i_lpBuffer)
		{
			S32 value = *(const S32*)i_lpBuffer;

			this->SetValue(value);

			return sizeof(S32);
		}
	};

	/** 設定項目(実数)を作成する */
	extern "C" GRAVISBELL_SETTINGDATA_API IItem_Int* CreateItem_Int(const wchar_t i_szID[], const wchar_t i_szName[], const wchar_t i_szText[], S32 minValue, S32 maxValue, S32 defaultValue)
	{
		return new Item_Int(i_szID, i_szName, i_szText, minValue, maxValue, defaultValue);
	}

}	// Standard
}	// SettingData
}	// Gravisbell