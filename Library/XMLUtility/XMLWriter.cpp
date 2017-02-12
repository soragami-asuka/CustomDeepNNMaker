//==================================
// XML書き込み用クラス
//==================================
#include"stdafx.h"

#pragma warning(disable:4996)

#include"XMLWriter.h"

#include"../Utility/StringUtility.h"

#include<./boost/lexical_cast.hpp>
#include<./boost/uuid/uuid_io.hpp>

using namespace XMLUtility;


/** コンストラクタ */
XMLWriter::XMLWriter()
	:	pWriter	(NULL)
{
}
/** デストラクタ */
XMLWriter::~XMLWriter()
{
	if(this->pWriter)
	{
		//======================================
		// 自動的に属性が閉じられる
		//======================================
		if(FAILED(pWriter->WriteEndDocument())){
			return;
		}

		//======================================
		// 終了
		//======================================
		if(FAILED(pWriter->Flush())){
			return;
		}
	}
}

/** 初期化 */
LONG XMLWriter::Initialize(const std::wstring& filePath)
{
	if(FAILED(CreateXmlWriter(__uuidof(::IXmlWriter), reinterpret_cast<void**>(&pWriter), 0))){
		return -1;
	}

	// XMLファイルパス作成
	TCHAR xml[MAX_PATH];
	GetModuleFileName(NULL, xml, sizeof(xml) / sizeof(TCHAR));
	PathRemoveFileSpec(xml);
	PathAppend(xml, filePath.c_str());

	// ファイルストリーム作成
	CComPtr<IStream> pStream;
	if(FAILED(SHCreateStreamOnFile(xml, STGM_CREATE | STGM_WRITE, &pStream))){
		return -1;
	}

	if(FAILED(pWriter->SetOutput(pStream))){
		return -1;
	}

	// インデント有効化
	if(FAILED(pWriter->SetProperty(XmlWriterProperty_Indent, TRUE))){
		return -1;
	}

	// <?xml version="1.0" encoding="UTF-8"?>
	if(FAILED(pWriter->WriteStartDocument(XmlStandalone_Omit))){
		return -1;
	}

	return 0;
}



//=========================================
// 要素の書き込み
//=========================================
/** 要素を開始 */
LONG XMLWriter::StartElement(const std::string&  name)
{
	return StartElement(Utility::ShiftjisToUnicode(name));
}
LONG XMLWriter::StartElement(const std::wstring& name)
{
	if(FAILED(pWriter->WriteStartElement(NULL, name.c_str(), NULL))){
		return -1;
	}
	return 0;
}

/** 要素を終了 */
LONG XMLWriter::EndElement()
{
	// 属性を終了
	if(FAILED(pWriter->WriteFullEndElement()))
		return -1;
	return 0;
}


/** 要素に文字列を書き込み */
LONG XMLWriter::WriteElement(const std::string&  name, const std::string& value)
{
	return WriteElement(Utility::ShiftjisToUnicode(name), Utility::ShiftjisToUnicode(value));
}
LONG XMLWriter::WriteElement(const std::wstring&  name, const std::wstring& value)
{
	// 要素の開始
	if(StartElement(name) != 0)
		return -1;

	// 値の書き込み
	if(pWriter->WriteString(value.c_str()) != 0)
		return -1;

	// 要素の終了
	if(EndElement() != 0)
		return -1;

	return 0;
}

/** 要素に文字列を書き込み */
LONG XMLWriter::WriteElement(const std::string&  name, const char value[])
{
	return WriteElement(name, (std::string)value);
}
LONG XMLWriter::WriteElement(const std::wstring&  name, const WCHAR value[])
{
	return WriteElement(name, (std::wstring)value);
}

/** 要素に整数(32bit)を書き込み */
LONG XMLWriter::WriteElement(const std::string&  name, __int32 value)
{
	return WriteElement(Utility::ShiftjisToUnicode(name), value);
}
LONG XMLWriter::WriteElement(const std::wstring&  name, __int32 value)
{
	WCHAR szBuf[256];
	swprintf(szBuf, L"%d", value);
	return WriteElement(name, (std::wstring)szBuf);
}
/** 要素に整数(64bit)を書き込み */
LONG XMLWriter::WriteElement(const std::string&  name, __int64 value)
{
	return WriteElement(Utility::ShiftjisToUnicode(name), value);
}
LONG XMLWriter::WriteElement(const std::wstring&  name, __int64 value)
{
	WCHAR szBuf[256];
	swprintf(szBuf, L"%lld", value);
	return WriteElement(name, (std::wstring)szBuf);
}

/** 要素に符号無し整数を書き込み */
LONG XMLWriter::WriteElement(const std::string&  name, unsigned long value)
{
	return WriteElement(Utility::ShiftjisToUnicode(name), value);
}
LONG XMLWriter::WriteElement(const std::wstring&  name, unsigned long value)
{
	WCHAR szBuf[256];
	swprintf(szBuf, L"%ul", value);
	return WriteElement(name, (std::wstring)szBuf);
}

/** 要素に実数を書き込み */
LONG XMLWriter::WriteElement(const std::string&  name, double value)
{
	return WriteElement(Utility::ShiftjisToUnicode(name), value);
}
LONG XMLWriter::WriteElement(const std::wstring&  name, double value)
{
	WCHAR szBuf[256];
	swprintf(szBuf, L"%f", value);
	return WriteElement(name, (std::wstring)szBuf);
}

/** 要素に論理値を書き込み */
LONG XMLWriter::WriteElement(const std::string&  name, bool value)
{
	return WriteElement(Utility::ShiftjisToUnicode(name), value);
}
LONG XMLWriter::WriteElement(const std::wstring&  name, bool value)
{
	if(value)
		return WriteElement(name, L"true");
	else
		return WriteElement(name, L"false");
}

/** 要素にGUIDを書き込み */
LONG XMLWriter::WriteElement(const std::string&  name, const boost::uuids::uuid& value)
{
	return this->WriteElement(Utility::ShiftjisToUnicode(name), value);
}
LONG XMLWriter::WriteElement(const std::wstring&  name, const boost::uuids::uuid& value)
{
	return this->WriteElement(name, boost::lexical_cast<std::wstring>(value));
}


//=======================================
// 要素に文字列を追加
//=======================================
/** 要素に文字列を書き込み */
LONG XMLWriter::AddElementString(const std::string& value)
{
	return this->AddElementString(Utility::ShiftjisToUnicode(value));
}
LONG XMLWriter::AddElementString(const std::wstring& value)
{
	// 値の書き込み
	if(pWriter->WriteString(value.c_str()) != 0)
		return -1;
	return 0;
}
/** 要素に文字列を書き込み */
LONG XMLWriter::AddElementString(const char value[])
{
	return this->AddElementString((std::string)value);
}
LONG XMLWriter::AddElementString(const WCHAR value[])
{
	return this->AddElementString((std::wstring)value);
}
/** 要素に整数を書き込み */
LONG XMLWriter::AddElementString(__int32 value)
{
	WCHAR szBuf[256];
	swprintf(szBuf, L"%d", value);
	return this->AddElementString((std::wstring)szBuf);
}
/** 要素に整数(63bit)を書き込み */
LONG XMLWriter::AddElementString(__int64 value)
{
	WCHAR szBuf[256];
	swprintf(szBuf, L"%lld", value);
	return this->AddElementString((std::wstring)szBuf);
}
/** 要素に符号無し整数を書き込み */
LONG XMLWriter::AddElementString(unsigned long value)
{
	WCHAR szBuf[256];
	swprintf(szBuf, L"%ul", value);
	return this->AddElementString((std::wstring)szBuf);
}
/** 要素に実数を書き込み */
LONG XMLWriter::AddElementString(double value)
{
	WCHAR szBuf[256];
	swprintf(szBuf, L"%f", value);
	return this->AddElementString((std::wstring)szBuf);
}
/** 要素に論理値を書き込み */
LONG XMLWriter::AddElementString(bool value)
{
	if(value)
		return this->AddElementString(L"true");
	else
		return this->AddElementString(L"false");
}
/** 要素にGUIDを書き込み */
LONG XMLWriter::AddElementString(const boost::uuids::uuid& value)
{
	return this->AddElementString(boost::lexical_cast<std::wstring>(value));
}



//=======================================
// 要素に属性を追加
//=======================================
/** 要素に属性を追加 */
LONG XMLWriter::AddAttribute(const std::string& id, const std::string& value)
{
	return AddAttribute(Utility::ShiftjisToUnicode(id), Utility::ShiftjisToUnicode(value));
}
LONG XMLWriter::AddAttribute(const std::wstring& id, const std::wstring& value)
{
	if(FAILED(pWriter->WriteAttributeString(NULL, id.c_str(), NULL, value.c_str())) )
		return -1;
	return 0;
}

/** 要素に属性を追加(文字列) */
LONG XMLWriter::AddAttribute(const std::string& id, const char value[])
{
	return AddAttribute(Utility::ShiftjisToUnicode(id), Utility::ShiftjisToUnicode(value));
}
LONG XMLWriter::AddAttribute(const std::wstring& id, const WCHAR value[])
{
	return AddAttribute(id, (std::wstring)value);
}

/** 要素に属性を追加(整数32bit) */
LONG XMLWriter::AddAttribute(const std::string& id, int value)
{
	return AddAttribute(Utility::ShiftjisToUnicode(id), value);
}
LONG XMLWriter::AddAttribute(const std::wstring& id, int value)
{
	WCHAR szBuf[256];
	swprintf(szBuf, L"%d", value);
	return AddAttribute(id, (std::wstring)szBuf);
}

/** 要素に属性を追加(整数64bit) */
LONG XMLWriter::AddAttribute(const std::string& id, __int64 value)
{
	return AddAttribute(Utility::ShiftjisToUnicode(id), value);
}
LONG XMLWriter::AddAttribute(const std::wstring& id, __int64 value)
{
	WCHAR szBuf[256];
	swprintf(szBuf, L"%lld", value);
	return AddAttribute(id, (std::wstring)szBuf);
}

/** 要素に属性を追加(実数) */
LONG XMLWriter::AddAttribute(const std::string& id, double value)
{
	return AddAttribute(Utility::ShiftjisToUnicode(id), value);
}
LONG XMLWriter::AddAttribute(const std::wstring& id, double value)
{
	WCHAR szBuf[256];
	swprintf(szBuf, L"%f", value);
	return AddAttribute(id, (std::wstring)szBuf);
}

/** 要素に属性を追加(論理値) */
LONG XMLWriter::AddAttribute(const std::string& id, bool value)
{
	return this->AddAttribute(Utility::ShiftjisToUnicode(id), value);
}
LONG XMLWriter::AddAttribute(const std::wstring& id, bool value)
{
	if(value)
		return this->AddAttribute(id, L"true");
	else
		return this->AddAttribute(id, L"false");
}

/** 要素に属性を追加する(GUID) */
LONG XMLWriter::AddAttribute(const std::string& id, const boost::uuids::uuid& value)
{
	return this->AddAttribute(Utility::ShiftjisToUnicode(id), value);
}
LONG XMLWriter::AddAttribute(const std::wstring& id, const boost::uuids::uuid& value)
{
	return this->AddAttribute(id, boost::lexical_cast<std::wstring>(value));
}