//====================================
// データフォーマット定義の本体情報
//====================================
#ifndef __GRAVISBELL_DATAFORMAT_BINARY_DATAFORMAT_CLASS_H__
#define __GRAVISBELL_DATAFORMAT_BINARY_DATAFORMAT_CLASS_H__


#include "stdafx.h"


#include"Library/DataFormat/Binary.h"

#include<string>
#include<vector>
#include<list>
#include<set>
#include<map>

#include<boost/property_tree/ptree.hpp>
#include<boost/property_tree/xml_parser.hpp>
#include <boost/iostreams/device/file_descriptor.hpp>
#include <boost/iostreams/stream.hpp>

#include"Library/Common/StringUtility/StringUtility.h"

#include"DataFormatItem.h"

namespace Gravisbell {
namespace DataFormat {
namespace Binary {

	class CDataFormat;

	/** 文字列を値に変換.進数判定付き */
	S32 ConvertString2Int(const std::wstring& buf);
	/** 文字列を値に変換.進数判定付き */
	U32 ConvertString2UInt(const std::wstring& buf);
	/** 文字列を値に変換.進数判定付き */
	F32 ConvertString2Float(const std::wstring& buf);

	/** データ構造 */
	struct DataStruct
	{
		std::wstring m_x;
		std::wstring m_y;
		std::wstring m_z;
		std::wstring m_ch;

		F32 m_false;
		F32 m_true;
	};
	/** データ本体 */
	struct DataInfo
	{
		DataStruct dataStruct;	/**< データ構造 */
		std::vector<F32*> lpData;	/**< 実データ */

		/** コンストラクタ */
		DataInfo();
		/** デストラクタ */
		~DataInfo();

		/** データ数を取得 */
		U32 GetDataCount()const;
	};


	/** データフォーマット */
	class CDataFormat : public IDataFormat
	{
	private:
		std::wstring name;	/**< 名前 */
		std::wstring text;	/**< 説明文 */

		std::map<std::wstring, S32> lpVariable;	/**< 変数と現在変数が取っている値. */
		std::map<std::wstring, DataInfo> lpData;			/**< データ一覧 */

		std::list<Format::CItem_base*> lpDataFormat;

		bool onReverseByteOrder;

	public:
		/** コンストラクタ */
		CDataFormat();
		/** コンストラクタ */
		CDataFormat(const wchar_t i_szName[], const wchar_t i_szText[], bool onReverseByteOrder);
		/** デストラクタ */
		virtual ~CDataFormat();

	public:
		/** 名前の取得 */
		const wchar_t* GetName()const;
		/** 説明文の取得 */
		const wchar_t* GetText()const;

		/** X次元の要素数を取得 */
		U32 GetBufferCountX(const wchar_t i_szCategory[])const;
		/** Y次元の要素数を取得 */
		U32 GetBufferCountY(const wchar_t i_szCategory[])const;
		/** Z次元の要素数を取得 */
		U32 GetBufferCountZ(const wchar_t i_szCategory[])const;
		/** CH次元の要素数を取得 */
		U32 GetBufferCountCH(const wchar_t i_szCategory[])const;

		/** データ構造を取得 */
		IODataStruct GetDataStruct(const wchar_t i_szCategory[])const;

		/** データ情報を取得 */
		const DataInfo* GetDataInfo(const wchar_t i_szCategory[])const;
		/** データ情報を取得 */
		DataInfo* GetDataInfo(const wchar_t i_szCategory[]);

		/** データ情報を追加する */
		Gravisbell::ErrorCode AddDataInfo(const wchar_t i_szCategory[], const wchar_t i_x[], const wchar_t i_y[], const wchar_t i_z[], const wchar_t i_ch[], F32 i_false, F32 i_true);
		/** データ情報に値を書き込む */
		Gravisbell::ErrorCode SetDataValue(const wchar_t i_szCategory[], U32 i_No, U32 i_x, U32 i_y, U32 i_z, U32 i_ch, F32 value);
		/** データ情報に値を書き込む.
			0.0=false.
			1.0=true.
			として値を格納する */
		Gravisbell::ErrorCode SetDataValueNormalize(const wchar_t i_szCategory[], U32 i_No, U32 i_x, U32 i_y, U32 i_z, U32 i_ch, F32 value);
		/** データ情報に値を書き込む */
		Gravisbell::ErrorCode SetDataValue(const wchar_t i_szCategory[], U32 i_No, U32 i_x, U32 i_y, U32 i_z, U32 i_ch, bool value);

		/** ID指定で変数の値を取得する.(直値判定付き) */
		S32 GetVariableValue(const wchar_t i_szID[])const;
		/** ID指定で変数の値を取得する.(直値判定付き.)(float型として値を返す) */
		F32 GetVariableValueAsFloat(const wchar_t i_szID[])const;
		/** ID指定で変数に値を設定する.(直値判定付き) */
		void SetVariableValue(const wchar_t i_szID[], S32 value);

		/** カテゴリー数を取得する */
		U32 GetCategoryCount()const;
		/** カテゴリー名を番号指定で取得する */
		const wchar_t* GetCategoryNameByNum(U32 categoryNo)const;

		/** Byte-Orderの反転フラグを取得する */
		bool GetOnReverseByteOrder()const;

	public:
		/** データ数を取得する */
		U32 GetDataCount()const;

		/** データを取得する */
		const F32* GetDataByNum(U32 i_dataNo, const wchar_t i_szCategory[])const;

	public:
		/** 正規化処理.
			データの追加が終了した後、一度のみ実行. 複数回実行すると値がおかしくなるので注意. */
		Gravisbell::ErrorCode Normalize();


	public:
		/** データフォーマット数を取得する */
		U32 GetDataFormatCount()const;

		/** データフォーマットを全削除する */
		Gravisbell::ErrorCode ClearDataFormat();

		/** データフォーマットを追加する */
		Gravisbell::ErrorCode AddDataFormat(Format::CItem_base* pDataFormat);


	public:
		/** バイナリデータを読み込む.
			@param	i_lpBuf		バイナリ先頭アドレス.
			@param	i_byteCount	読込可能なバイト数.
			@return	実際に読み込んだバイト数. 失敗した場合は負の値 */
		S32 LoadBinary(const BYTE* i_lpBuf, U32 i_byteCount);
	};


}	// Binary
}	// DataFormat
}	// Gravisbell


#endif	// __GRAVISBELL_DATAFORMAT_BINARY_DATAFORMAT_CLASS_H__