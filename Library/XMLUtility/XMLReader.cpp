//==================================
// XML読み込み用クラス
//==================================
#include"stdafx.h"

#pragma warning(disable:4996)

#include"XMLReader.h"

#include"../Utility/StringUtility.h"
#include <atlstr.h>   // CStringを使用するため

#include<./boost/uuid/uuid_generators.hpp>

using namespace XMLUtility;


/** コンストラクタ */
XMLReader::XMLReader()
	:	pReader	(NULL)
{
}
/** デストラクタ */
XMLReader::~XMLReader()
{
}


/** 初期化 */
LONG XMLReader::Initialize(const std::wstring& filePath)
{
	//===================================
	// 開始処理
	//===================================
	if(FAILED(CreateXmlReader(__uuidof(::IXmlReader), reinterpret_cast<void**>(&pReader), 0))){
		return -1;
	}

	// XMLファイルパス作成
	TCHAR xml[MAX_PATH];
	GetModuleFileName(NULL, xml, sizeof(xml) / sizeof(TCHAR));
	PathRemoveFileSpec(xml);
	PathAppend(xml, filePath.c_str());

	// ファイルストリーム作成
	CComPtr<IStream> pStream;
	if(FAILED(SHCreateStreamOnFile(xml, STGM_READ, &pStream))){
		return -1;
	}

	if(FAILED(pReader->SetInput(pStream))){
		return -1;
	}

	return 0;
}


/** 次のノードを読み込む */
LONG XMLReader::Read(NodeType& nodeType)
{
	XmlNodeType tmpNodeType;

	if(this->pReader->Read(&tmpNodeType) == S_OK)
	{
		nodeType = (NodeType)tmpNodeType;
		return 0;
	}
	return -1;
}

/** 現在の属性名を取得する */
std::string XMLReader::GetElementName()
{
	return Utility::UnicodeToShiftjis(GetElementNameW());
}
std::wstring XMLReader::GetElementNameW()
{
	LPCWSTR pwszLocalName;
	if(FAILED(pReader->GetLocalName(&pwszLocalName, NULL))){
		return L"";
	}
	return static_cast<LPCTSTR>(CString(pwszLocalName));	//文字列に変換
}


/** 値を文字列で取得する */
std::string XMLReader::ReadElementValueString()
{
	return Utility::UnicodeToShiftjis(ReadElementValueWString());
}
std::wstring XMLReader::ReadElementValueWString()
{
	std::wstring strBuf = L"";
	int dips = 0;
			
	if(pReader->IsEmptyElement())
	{
		return L"";
	}

	//読み込み
	CString strElement = "";
	XmlNodeType nodeType;
	while(S_OK == pReader->Read(&nodeType))
	{
		switch(nodeType)
		{
		case XmlNodeType_Element:	//属性の開始
			{
				dips++;
			}
			break;

		case XmlNodeType_EndElement:	//属性の終了
			{
				if(dips == 0)
					return strBuf;

				dips--;
			}
			break;
			
		case XmlNodeType_Text:	//文字列の読み込み
			{
				CString value;
				const UINT buffSize = 1024;  //バッファサイズ
				WCHAR buff[buffSize];
				UINT charsRead;
				HRESULT hr = pReader->ReadValueChunk(buff, buffSize - 1, &charsRead);
				if(hr == S_OK)
				{
					buff[charsRead] = L'\0';
					value = buff;
				}

				strBuf = value;
			}
			break;
		}
	}
	return L"";
}

/** 値を10進整数で取得する */
__int32 XMLReader::ReadElementValueInt32()
{
	return atoi(ReadElementValueString().c_str());
}
__int64 XMLReader::ReadElementValueInt64()
{
	return _atoi64(ReadElementValueString().c_str());
}
/** 値を16進整数で取得する */
__int32 XMLReader::ReadElementValueIntX32()
{
	return strtoul(ReadElementValueString().c_str(), NULL, 16);
}
__int64 XMLReader::ReadElementValueIntX64()
{
	return strtoul(ReadElementValueString().c_str(), NULL, 16);
}
/** 値を実数で取得する */
double XMLReader::ReadElementValueDouble()
{
	return atof(ReadElementValueString().c_str());
}
/** 値を論理値で取得する */
bool XMLReader::ReadElementValueBool()
{
	std::string buf = ReadElementValueString();
	if(buf == "true")
		return true;
	else
		return false;
}
/** 値をGUIDで取得する */
boost::uuids::uuid XMLReader::ReadElementValueGUID()
{
	return boost::uuids::string_generator()(this->ReadElementValueWString());
}


/** 値を文字列配列の番号で取得する */
LONG XMLReader::ReadElementValueEnum(const std::string lpName[], LONG valueCount)
{
	std::string value = ReadElementValueString();
	for(LONG i=0; i<valueCount; i++)
	{
		if(lpName[i] == value)
			return i;
	}
	return -1;
}
LONG XMLReader::ReadElementValueEnum(const std::wstring lpName[], LONG valueCount)
{
	std::wstring value = ReadElementValueWString();
	for(LONG i=0; i<valueCount; i++)
	{
		if(lpName[i] == value)
			return i;
	}
	return -1;
}

/** 値を文字列配列の番号で取得する */
LONG XMLReader::ReadElementValueEnum(const std::vector<std::string>& lpName)
{
	return ReadElementValueEnum(&lpName[0], lpName.size());
}
LONG XMLReader::ReadElementValueEnum(const std::vector<std::wstring>& lpName)
{
	return ReadElementValueEnum(&lpName[0], lpName.size());
}


/** 属性をリストで取得する */
std::map<std::string, std::string> XMLReader::ReadAttributeList()
{
	std::map<std::wstring, std::wstring> lpAttributeW = this->ReadAttributeListW();
	std::map<std::string, std::string> lpAttribute;

	for(auto it : lpAttributeW)
	{
		lpAttribute[Utility::UnicodeToShiftjis(it.first)] = Utility::UnicodeToShiftjis(it.second);
	}

	return lpAttribute;
}
std::map<std::wstring, std::wstring> XMLReader::ReadAttributeListW()
{
	std::map<std::wstring, std::wstring> lpAttriBute;

	HRESULT hr = pReader->MoveToFirstAttribute();
	while(hr == S_OK)
	{
		LPCWSTR pwszAttributeName;
		LPCWSTR pwszAttributeValue;
		if( FAILED(pReader->GetLocalName(&pwszAttributeName, NULL)) )
			return lpAttriBute;
		if( FAILED(pReader->GetValue(&pwszAttributeValue, NULL)) )
			return lpAttriBute;

		lpAttriBute[pwszAttributeName] = pwszAttributeValue;

		hr = pReader->MoveToNextAttribute();	//次の属性へ
	}
	return lpAttriBute;
}
