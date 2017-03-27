// DataFormatStringArray.cpp : DLL アプリケーション用にエクスポートされる関数を定義します。
//

#include "stdafx.h"


#include"DataFormat.h"

#include<string>
#include<vector>
#include<set>
#include<map>


namespace Gravisbell {
namespace DataFormat {
namespace StringArray {
	
	struct BoolValue
	{
		F32 trueValue;	/**< trueの時の値 */
		F32 falseValue;	/**< falseの時の値 */

		BoolValue()
			:	trueValue	(1.0f)
			,	falseValue	(0.0f)
		{
		}
		BoolValue(F32 trueValue, F32 falseValue)
			:	trueValue	(trueValue)
			,	falseValue	(falseValue)
		{
		}
	};

	/** データフォーマットのアイテム */
	class IDataFormatItem
	{
	public:
		/** コンストラクタ */
		IDataFormatItem(){}
		/** デストラクタ */
		virtual ~IDataFormatItem(){}

	public:
		/** 使用バッファ数を返す */
		virtual U32 GetBufferCount()const = 0;

		/** データを追加する */
		virtual U32 AddValue(const std::wstring& value) = 0;

		/** 正規化 */
		virtual ErrorCode Normalize() = 0;
	};


	/** データフォーマット */
	class CDataFormat : public IDataFormat
	{
	private:
		std::wstring name;	/**< 名前 */
		std::wstring text;	/**< 説明文 */

		std::set<std::wstring> lpCategoryName;	/**< データ種別名一覧 */

		std::map<std::wstring, BoolValue>	lpBoolValue;	/**< bool値をF32に変換する設定値の一覧.	<データ種別名, 変換データ> */

	public:
		/** コンストラクタ */
		CDataFormat()
		:	CDataFormat(L"", L"")
		{
		}
		/** コンストラクタ */
		CDataFormat(const wchar_t i_szName[], const wchar_t i_szText[])
		:	name	(i_szName)
		,	text	(i_szText)
		{
			// デフォルト値を設定する
			lpBoolValue[L""] = BoolValue();
		}
		/** デストラクタ */
		virtual ~CDataFormat(){}

	public:
		/** 名前の取得 */
		const wchar_t* GetName()const
		{
			return name.c_str();
		}
		/** 説明文の取得 */
		const wchar_t* GetText()const
		{
			return text.c_str();
		}

		/** X次元の要素数を取得 */
		U32 GetBufferCountX(const wchar_t i_szCategory[])const
		{
			return 1;
		}
		/** Y次元の要素数を取得 */
		U32 GetBufferCountY(const wchar_t i_szCategory[])const
		{
			return 1;
		}
		/** Z次元の要素数を取得 */
		U32 GetBufferCountZ(const wchar_t i_szCategory[])const
		{
			return 1;
		}
		/** CH次元の要素数を取得 */
		U32 GetBufferCountCH(const wchar_t i_szCategory[])const
		{
			return 0;
		}

		/** データ構造を取得 */
		IODataStruct GetDataStruct(const wchar_t i_szCategory[])const
		{
			return IODataStruct(this->GetBufferCountCH(i_szCategory), this->GetBufferCountX(i_szCategory), this->GetBufferCountY(i_szCategory), this->GetBufferCountZ(i_szCategory));
		}

		/** カテゴリー数を取得する */
		U32 GetCategoryCount()const
		{
			return this->lpCategoryName.size();
		}
		/** カテゴリー名を番号指定で取得する */
		const wchar_t* GetCategoryNameByNum(U32 categoryNo)const
		{
			if(categoryNo >= this->lpCategoryName.size())
				return NULL;

			auto it = this->lpCategoryName.begin();
			for(U32 no=0; no<categoryNo; no++)
				it++;
			return it->c_str();
		}

	public:
		/** trueの場合の値設定を取得する */
		F32 GetTrueValue(const wchar_t i_szCategory[] = L"")const
		{
			// カテゴリを検索
			auto it = this->lpBoolValue.find(i_szCategory);
			if(it != this->lpBoolValue.end())
				return it->second.trueValue;

			// デフォルト値が設定されていない場合は1.0を返す
			if((std::wstring)i_szCategory == L"")
				return 1.0f;

			// デフォルト値を再検索
			return GetTrueValue(L"");
		}
		/** falseの場合の値設定を取得する */
		F32 GetFalseValue(const wchar_t i_szCategory[] = L"")const
		{
			// カテゴリを検索
			auto it = this->lpBoolValue.find(i_szCategory);
			if(it != this->lpBoolValue.end())
				return it->second.falseValue;

			// デフォルト値が設定されていない場合は0.0を返す
			if((std::wstring)i_szCategory == L"")
				return 0.0f;

			// デフォルト値を再検索
			return GetFalseValue(L"");
		}

	};

	/** 文字列の配列を読み込むデータフォーマットを作成する */
	GRAVISBELL_LIBRAY_API IDataFormat* CreateDataFormat(const wchar_t i_szName[], const wchar_t i_szText[])
	{
		return new CDataFormat(i_szName, i_szText);
	}
	/** 文字列の配列を読み込むデータフォーマットを作成する */
	extern GRAVISBELL_LIBRAY_API IDataFormat* CreateDataFormatFromXML(const wchar_t szXMLFilePath[])
	{
		CDataFormat* pDataFormat = new CDataFormat();

		return pDataFormat;
	}


}	// StringArray
}	// DataFormat
}	// Gravisbell


