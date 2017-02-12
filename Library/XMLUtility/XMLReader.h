//==================================
// XML読み込み用クラス
//==================================
#pragma once

#include<Windows.h>
#include"XMLUtility.h"

#include<atlbase.h>  // CComPtrを使用するため
#include<xmllite.h>

namespace XMLUtility
{
	class XMLReader : public XMLUtility::IXmlReader
	{
	private:
		CComPtr<::IXmlReader> pReader;

	public:
		/** コンストラクタ */
		XMLReader();
		/** デストラクタ */
		virtual ~XMLReader();

	public:
		/** 初期化 */
		LONG Initialize(const std::wstring& filePath);

	public:
		/** 次のノードを読み込む */
		LONG Read(NodeType& nodeType);

		/** 現在の要素名を取得する */
		std::string GetElementName();
		std::wstring GetElementNameW();


		/** 値を文字列で取得する */
		std::string ReadElementValueString();
		std::wstring ReadElementValueWString();

		/** 値を10進整数で取得する */
		__int32 ReadElementValueInt32();
		__int64 ReadElementValueInt64();
		/** 値を16進整数で取得する */
		__int32 ReadElementValueIntX32();
		__int64 ReadElementValueIntX64();
		/** 値を実数で取得する */
		double ReadElementValueDouble();
		/** 値を論理値で取得する */
		bool ReadElementValueBool();
		/** 値をGUIDで取得する */
		boost::uuids::uuid ReadElementValueGUID();


		/** 値を文字列配列の番号で取得する */
		LONG ReadElementValueEnum(const std::string lpName[], LONG valueCount);
		LONG ReadElementValueEnum(const std::wstring lpName[], LONG valueCount);

		/** 値を文字列配列の番号で取得する */
		LONG ReadElementValueEnum(const std::vector<std::string>& lpName);
		LONG ReadElementValueEnum(const std::vector<std::wstring>& lpName);

		/** 属性をリストで取得する */
		std::map<std::string, std::string> ReadAttributeList();
		std::map<std::wstring, std::wstring> ReadAttributeListW();
	};
}