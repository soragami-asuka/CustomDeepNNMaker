//==================================
// XML�ǂݍ��ݗp�N���X
//==================================
#include"stdafx.h"

#pragma warning(disable:4996)

#include"XMLReader.h"

#include"../Utility/StringUtility.h"
#include <atlstr.h>   // CString���g�p���邽��

#include<./boost/uuid/uuid_generators.hpp>

using namespace XMLUtility;


/** �R���X�g���N�^ */
XMLReader::XMLReader()
	:	pReader	(NULL)
{
}
/** �f�X�g���N�^ */
XMLReader::~XMLReader()
{
}


/** ������ */
LONG XMLReader::Initialize(const std::wstring& filePath)
{
	//===================================
	// �J�n����
	//===================================
	if(FAILED(CreateXmlReader(__uuidof(::IXmlReader), reinterpret_cast<void**>(&pReader), 0))){
		return -1;
	}

	// XML�t�@�C���p�X�쐬
	TCHAR xml[MAX_PATH];
	GetModuleFileName(NULL, xml, sizeof(xml) / sizeof(TCHAR));
	PathRemoveFileSpec(xml);
	PathAppend(xml, filePath.c_str());

	// �t�@�C���X�g���[���쐬
	CComPtr<IStream> pStream;
	if(FAILED(SHCreateStreamOnFile(xml, STGM_READ, &pStream))){
		return -1;
	}

	if(FAILED(pReader->SetInput(pStream))){
		return -1;
	}

	return 0;
}


/** ���̃m�[�h��ǂݍ��� */
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

/** ���݂̑��������擾���� */
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
	return static_cast<LPCTSTR>(CString(pwszLocalName));	//������ɕϊ�
}


/** �l�𕶎���Ŏ擾���� */
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

	//�ǂݍ���
	CString strElement = "";
	XmlNodeType nodeType;
	while(S_OK == pReader->Read(&nodeType))
	{
		switch(nodeType)
		{
		case XmlNodeType_Element:	//�����̊J�n
			{
				dips++;
			}
			break;

		case XmlNodeType_EndElement:	//�����̏I��
			{
				if(dips == 0)
					return strBuf;

				dips--;
			}
			break;
			
		case XmlNodeType_Text:	//������̓ǂݍ���
			{
				CString value;
				const UINT buffSize = 1024;  //�o�b�t�@�T�C�Y
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

/** �l��10�i�����Ŏ擾���� */
__int32 XMLReader::ReadElementValueInt32()
{
	return atoi(ReadElementValueString().c_str());
}
__int64 XMLReader::ReadElementValueInt64()
{
	return _atoi64(ReadElementValueString().c_str());
}
/** �l��16�i�����Ŏ擾���� */
__int32 XMLReader::ReadElementValueIntX32()
{
	return strtoul(ReadElementValueString().c_str(), NULL, 16);
}
__int64 XMLReader::ReadElementValueIntX64()
{
	return strtoul(ReadElementValueString().c_str(), NULL, 16);
}
/** �l�������Ŏ擾���� */
double XMLReader::ReadElementValueDouble()
{
	return atof(ReadElementValueString().c_str());
}
/** �l��_���l�Ŏ擾���� */
bool XMLReader::ReadElementValueBool()
{
	std::string buf = ReadElementValueString();
	if(buf == "true")
		return true;
	else
		return false;
}
/** �l��GUID�Ŏ擾���� */
boost::uuids::uuid XMLReader::ReadElementValueGUID()
{
	return boost::uuids::string_generator()(this->ReadElementValueWString());
}


/** �l�𕶎���z��̔ԍ��Ŏ擾���� */
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

/** �l�𕶎���z��̔ԍ��Ŏ擾���� */
LONG XMLReader::ReadElementValueEnum(const std::vector<std::string>& lpName)
{
	return ReadElementValueEnum(&lpName[0], lpName.size());
}
LONG XMLReader::ReadElementValueEnum(const std::vector<std::wstring>& lpName)
{
	return ReadElementValueEnum(&lpName[0], lpName.size());
}


/** ���������X�g�Ŏ擾���� */
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

		hr = pReader->MoveToNextAttribute();	//���̑�����
	}
	return lpAttriBute;
}
