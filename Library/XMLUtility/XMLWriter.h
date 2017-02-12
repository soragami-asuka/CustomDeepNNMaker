//==================================
// XML書き込み用クラス
//==================================
#pragma once

#include<Windows.h>
#include"XMLUtility.h"

#include<atlbase.h>  // CComPtrを使用するため
#include<xmllite.h>

namespace XMLUtility
{
	class XMLWriter : public IXmlWriter
	{
	private:
		CComPtr<::IXmlWriter> pWriter;

	public:
		/** コンストラクタ */
		XMLWriter();
		/** デストラクタ */
		~XMLWriter();

	public:
		/** 初期化 */
		LONG Initialize(const std::wstring& filePath);

	public:
		/** 要素を開始 */
		LONG StartElement(const std::string&  name);
		LONG StartElement(const std::wstring& name);

		/** 要素を終了 */
		LONG EndElement();


	public:
		/** 要素に文字列を書き込み */
		LONG WriteElement(const std::string&  name, const std::string& value);
		LONG WriteElement(const std::wstring&  name, const std::wstring& value);
		/** 要素に文字列を書き込み */
		LONG WriteElement(const std::string&  name, const char value[]);
		LONG WriteElement(const std::wstring&  name, const WCHAR value[]);
		/** 要素に整数(32bit)を書き込み */
		LONG WriteElement(const std::string&  name, __int32 value);
		LONG WriteElement(const std::wstring&  name, __int32 value);
		/** 要素に整数(64bit)を書き込み */
		LONG WriteElement(const std::string&  name, __int64 value);
		LONG WriteElement(const std::wstring&  name, __int64 value);
		/** 要素に符号無し整数を書き込み */
		LONG WriteElement(const std::string&  name, unsigned long value);
		LONG WriteElement(const std::wstring&  name, unsigned long value);
		/** 要素に実数を書き込み */
		LONG WriteElement(const std::string&  name, double value);
		LONG WriteElement(const std::wstring&  name, double value);
		/** 要素に論理値を書き込み */
		LONG WriteElement(const std::string&  name, bool value);
		LONG WriteElement(const std::wstring&  name, bool value);
		/** 要素にGUIDを書き込み */
		LONG WriteElement(const std::string&  name, const boost::uuids::uuid& value);
		LONG WriteElement(const std::wstring&  name, const boost::uuids::uuid& value);


		/** 要素に文字列を書き込み */
		LONG AddElementString(const std::string& value);
		LONG AddElementString(const std::wstring& value);
		/** 要素に文字列を書き込み */
		LONG AddElementString(const char value[]);
		LONG AddElementString(const WCHAR value[]);
		/** 要素に整数(32bit)を書き込み */
		LONG AddElementString(__int32 value);
		/** 要素に整数(63bit)を書き込み */
		LONG AddElementString(__int64 value);
		/** 要素に符号無し整数を書き込み */
		LONG AddElementString(unsigned long value);
		/** 要素に実数を書き込み */
		LONG AddElementString(double value);
		/** 要素に論理値を書き込み */
		LONG AddElementString(bool value);
		/** 要素にGUIDを書き込み */
		LONG AddElementString(const boost::uuids::uuid& value);


		/** 要素に属性を追加 */
		LONG AddAttribute(const std::string& id, const std::string& value);
		LONG AddAttribute(const std::wstring& id, const std::wstring& value);
		/** 要素に属性を追加(文字列) */
		LONG AddAttribute(const std::string& id, const char value[]);
		LONG AddAttribute(const std::wstring& id, const WCHAR value[]);
		/** 要素に属性を追加(整数32bit) */
		LONG AddAttribute(const std::string& id, __int32 value);
		LONG AddAttribute(const std::wstring& id, __int32 value);
		/** 要素に属性を追加(整数64bit) */
		LONG AddAttribute(const std::string& id, __int64 value);
		LONG AddAttribute(const std::wstring& id, __int64 value);
		/** 要素に属性を追加(倍精度実数) */
		LONG AddAttribute(const std::string& id, double value);
		LONG AddAttribute(const std::wstring& id, double value);
		/** 要素に属性を追加(論理値) */
		LONG AddAttribute(const std::string& id, bool value);
		LONG AddAttribute(const std::wstring& id, bool value);
		/** 要素に属性を追加する(GUID) */
		LONG AddAttribute(const std::string& id, const boost::uuids::uuid& value);
		LONG AddAttribute(const std::wstring& id, const boost::uuids::uuid& value);
	};
}
