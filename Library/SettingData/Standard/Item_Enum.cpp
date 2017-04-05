//==================================
// 設定項目(列挙型)
//==================================
#include "stdafx.h"

#include"SettingData.h"
#include"IItemEx_Enum.h"
#include"ItemBase.h"

#include<string>
#include<vector>

namespace Gravisbell {
namespace SettingData {
namespace Standard {

	class Item_Enum : public ItemBase<IItemEx_Enum>
	{
	private:
		struct EnumItem
		{
			std::wstring id;
			std::wstring name;
			std::wstring text;

			/** コンストラクタ */
			EnumItem()
				: id (L"")
				, name (L"")
				, text (L"")
			{
			}
			/** コンストラクタ */
			EnumItem(const std::wstring& id, const std::wstring& name, const std::wstring& text)
				: id		(id)
				, name		(name)
				, text		(text)
			{
			}
			/** コピーコンストラクタ */
			EnumItem(const EnumItem& item)
				: id		(item.id)
				, name		(item.name)
				, text		(item.text)
			{
			}

			/** =演算 */
			const EnumItem& operator=(const EnumItem& item)
			{
				this->id = item.id;
				this->name = item.name;
				this->text = item.text;

				return *this;
			}


			bool operator==(const EnumItem& item)const
			{
				if(this->id != item.id)
					return false;
				if(this->name != item.name)
					return false;
				if(this->text != item.text)
					return false;
				return true;
			}
			bool operator!=(const EnumItem& item)const
			{
				return !(*this == item);
			}
		};

	private:
		std::vector<EnumItem> lpEnumItem;

		std::wstring value;
		std::wstring defaultValue;


	public:
		/** コンストラクタ */
		Item_Enum(const wchar_t i_szID[], const wchar_t i_szName[], const wchar_t i_szText[])
			: ItemBase(i_szID, i_szName, i_szText)
		{
		}
		/** コピーコンストラクタ */
		Item_Enum(const Item_Enum& item)
			: ItemBase(item)
			, lpEnumItem (item.lpEnumItem)
			, value (item.value)
			, defaultValue (item.defaultValue)
		{
		}
		/** デストラクタ */
		virtual ~Item_Enum(){}
		
		/** 一致演算 */
		bool operator==(const IItemBase& item)const
		{
			// 種別の確認
			if(this->GetItemType() != item.GetItemType())
				return false;

			// アイテムを変換
			const Item_Enum* pItem = dynamic_cast<const Item_Enum*>(&item);
			if(pItem == NULL)
				return false;

			if(ItemBase::operator!=(*pItem))
				return false;

			for(U32 itemNum=0; itemNum<this->lpEnumItem.size(); itemNum++)
			{
				if(this->lpEnumItem[itemNum] != pItem->lpEnumItem[itemNum])
					return false;
			}
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
			return new Item_Enum(*this);
		}

	public:
		/** 設定項目種別を取得する */
		ItemType GetItemType()const
		{
			return ITEMTYPE_ENUM;
		}

	public:
		/** 値を取得する */
		S32 GetValue()const
		{
			return this->GetNumByID(this->value.c_str());
		}

		/** 値を設定する
			@param value	設定する値
			@return 成功した場合0 */
		ErrorCode SetValue(S32 value)
		{
			if(value < 0)
				return ERROR_CODE_COMMON_OUT_OF_VALUERANGE;
			if(value >= (S32)this->lpEnumItem.size())
				return ERROR_CODE_COMMON_OUT_OF_VALUERANGE;

			this->value = this->lpEnumItem[value].id;

			return ERROR_CODE_NONE;
		}
		/** 値を設定する(文字列指定)
			@param i_szID	設定する値(文字列指定)
			@return 成功した場合0 */
		ErrorCode SetValue(const wchar_t i_szEnumID[])
		{
			return this->SetValue(this->GetNumByID(i_szEnumID));
		}

	public:
		/** 列挙要素数を取得する */
		U32 GetEnumCount()const
		{
			return this->lpEnumItem.size();
		}
		/** 列挙要素IDを番号指定で取得する.
			@param num		要素番号
			@param o_szBuf	格納先バッファ
			@return 成功した場合0 */
		S32 GetEnumID(S32 num, wchar_t o_szBuf[])const
		{
			if(num < 0)
				return -1;
			if(num >= (S32)this->lpEnumItem.size())
				return -1;
			
			wcscpy(o_szBuf, this->lpEnumItem[num].id.c_str());

			return 0;
		}
		/** 列挙要素IDを番号指定で取得する.
			@param num		要素番号
			@param o_szBuf	格納先バッファ
			@return 成功した場合0 */
		S32 GetEnumName(S32 num, wchar_t o_szBuf[])const
		{
			if(num < 0)
				return -1;
			if(num >= (S32)this->lpEnumItem.size())
				return -1;
			
			wcscpy(o_szBuf, this->lpEnumItem[num].name.c_str());

			return 0;
		}
		/** 列挙要素説明文を番号指定で取得する.
			@param num		要素番号
			@param o_szBuf	格納先バッファ
			@return 成功した場合0 */
		S32 GetEnumText(S32 num, wchar_t o_szBuf[])const
		{
			if(num < 0)
				return -1;
			if(num >= (S32)this->lpEnumItem.size())
				return -1;
			
			wcscpy(o_szBuf, this->lpEnumItem[num].text.c_str());

			return 0;
		}

		/** IDを指定して列挙要素番号を取得する
			@param i_szBuf　調べる列挙ID.
			@return 成功した場合0<=num<GetEnumCountの値. 失敗した場合は負の値が返る. */
		S32 GetNumByID(const wchar_t i_szEnumID[])const
		{
			std::wstring enumID = i_szEnumID;

			for(U32 itemNum=0; itemNum<this->lpEnumItem.size(); itemNum++)
			{
				if(this->lpEnumItem[itemNum].id == enumID)
					return (S32)itemNum;
			}

			return -1;
		}

		/** デフォルトの設定値を取得する */
		S32 GetDefault()const
		{
			return this->GetNumByID(this->defaultValue.c_str());
		}

	public:
		/** 列挙値を追加する.
			同一IDが既に存在する場合失敗する.
			@param szEnumID		追加するID
			@param szEnumName	追加する名前
			@param szComment	追加するコメント分.
			@return 成功した場合、追加されたアイテムの番号が返る. 失敗した場合は負の値が返る. */
		S32 AddEnumItem(const wchar_t szEnumID[], const wchar_t szEnumName[], const wchar_t szComment[])
		{
			// 同一IDが存在するか確認
			S32 sameID = this->GetNumByID(szEnumID);
			if(sameID >= 0)
				return -1;

			std::wstring id = szEnumID;
			if(id.size()+1 >= ID_BUFFER_MAX)
				return -1;

			// 追加
			this->lpEnumItem.push_back(EnumItem(id, szEnumName, szComment));

			return this->lpEnumItem.size()-1;
		}

		/** 列挙値を削除する.
			@param num	削除する列挙番号
			@return 成功した場合0 */
		S32 EraseEnumItem(S32 num)
		{
			if(num < 0)
				return -1;
			if(num >= (S32)this->lpEnumItem.size())
				return -1;

			// iteratorを進める
			auto it = this->lpEnumItem.begin();
			for(S32 i=0; i<num; i++)
				it++;

			// 削除
			this->lpEnumItem.erase(it);

			return 0;
		}
		/** 列挙値を削除する
			@param szEnumID 削除する列挙ID
			@return 成功した場合0 */
		S32 EraseEnumItem(const wchar_t szEnumID[])
		{
			return this->EraseEnumItem(this->GetNumByID(szEnumID));
		}

		/** デフォルト値を設定する.	番号指定.
			@param num デフォルト値に設定する番号. 
			@return 成功した場合0 */
		S32 SetDefaultItem(S32 num)
		{
			wchar_t szEnumID[ID_BUFFER_MAX];

			// IDを取得する
			if(this->GetEnumID(num, szEnumID) < 0)
				return -1;

			this->defaultValue = szEnumID;

			return 0;
		}
		/** デフォルト値を設定する.	ID指定. 
			@param szID デフォルト値に設定するID. 
			@return 成功した場合0 */
		S32 SetDefaultItem(const wchar_t szEnumID[])
		{
			if(this->GetNumByID(szEnumID) < 0)
				return -1;

			this->defaultValue = szEnumID;

			return 0;
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

			byteCount += sizeof(U32);		// 値のバッファサイズ
			byteCount += this->value.size();		// 値

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
			{
				// バッファサイズ
				U32 bufferSize = *(U32*)&i_lpBuffer[bufferPos];
				bufferPos += sizeof(U32);

				// 値
				std::vector<wchar_t> szBuf(bufferSize+1, NULL);
				for(U32 i=0; i<bufferSize; i++)
				{
					szBuf[i] = i_lpBuffer[bufferPos++];
				}
				std::wstring value = &szBuf[0];


				this->SetValue(value.c_str());
			}


			return bufferPos;
		}
		/** バッファに書き込む.
			@param o_lpBuffer	書き込み先バッファの先頭アドレス. GetUseBufferByteCountの戻り値のバイト数が必要
			@return 成功した場合書き込んだバッファサイズ.失敗した場合は負の値 */
		S32 WriteToBuffer(BYTE* o_lpBuffer)const
		{
			U32 bufferPos = 0;
			
			// 値
			{
				// バッファサイズ
				U32 bufferSize = this->value.size();;
				*(U32*)&o_lpBuffer[bufferPos] = bufferSize;
				bufferPos += sizeof(U32);

				// 値
				memcpy(&o_lpBuffer[bufferPos], this->value.c_str(), bufferSize);
				bufferPos += bufferSize;
			}


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
	
	/** 設定項目(列挙値)を作成する */
	extern "C" GRAVISBELL_SETTINGDATA_API IItemEx_Enum* CreateItem_Enum(const wchar_t i_szID[], const wchar_t i_szName[], const wchar_t i_szText[])
	{
		return new Item_Enum(i_szID, i_szName, i_szText);
	}

}	// Standard
}	// SettingData
}	// Gravisbell