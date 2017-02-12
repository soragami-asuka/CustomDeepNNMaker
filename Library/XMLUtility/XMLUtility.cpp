//==================================
// XML読み書き用クラス
//==================================
#include "stdafx.h"

#pragma warning(disable:4996)

#include "XMLUtility.h"

#include"../Utility/StringUtility.h"

#include"XMLWriter.h"
#include"XMLReader.h"

#pragma comment(lib, "xmllite.lib")


namespace XMLUtility
{
	/** XML書き込みクラスを作成する */
	XMLUTILITY_API XMLWriterPtr CreateXMLWriter(const std::string& filePath)
	{
		return CreateXMLWriter(Utility::ShiftjisToUnicode(filePath));
	}
	XMLUTILITY_API XMLWriterPtr CreateXMLWriter(const std::wstring& filePath)
	{
		XMLWriter* pWriter = new XMLWriter();
		if(pWriter->Initialize(filePath) == 0)
		{
			return XMLWriterPtr(pWriter);
		}

		return NULL;
	}

	/** XML読み込みクラスを作成する */
	XMLUTILITY_API XMLReaderPtr CreateXMLReader(const std::string& filePath)
	{
		return CreateXMLReader(Utility::ShiftjisToUnicode(filePath));
	}
	XMLUTILITY_API XMLReaderPtr CreateXMLReader(const std::wstring& filePath)
	{
		XMLReader* pReader = new XMLReader();
		if(pReader->Initialize(filePath) == 0)
		{
			return XMLReaderPtr(pReader);
		}

		return NULL;
	}
}