//==================================
// XML�������ݗp�N���X
//==================================
#include"stdafx.h"

#pragma warning(disable:4996)

#include"XMLWriter.h"

#include"../Utility/StringUtility.h"

#include<./boost/lexical_cast.hpp>
#include<./boost/uuid/uuid_io.hpp>

using namespace XMLUtility;


/** �R���X�g���N�^ */
XMLWriter::XMLWriter()
	:	pWriter	(NULL)
{
}
/** �f�X�g���N�^ */
XMLWriter::~XMLWriter()
{
	if(this->pWriter)
	{
		//======================================
		// �����I�ɑ�����������
		//======================================
		if(FAILED(pWriter->WriteEndDocument())){
			return;
		}

		//======================================
		// �I��
		//======================================
		if(FAILED(pWriter->Flush())){
			return;
		}
	}
}

/** ������ */
LONG XMLWriter::Initialize(const std::wstring& filePath)
{
	if(FAILED(CreateXmlWriter(__uuidof(::IXmlWriter), reinterpret_cast<void**>(&pWriter), 0))){
		return -1;
	}

	// XML�t�@�C���p�X�쐬
	TCHAR xml[MAX_PATH];
	GetModuleFileName(NULL, xml, sizeof(xml) / sizeof(TCHAR));
	PathRemoveFileSpec(xml);
	PathAppend(xml, filePath.c_str());

	// �t�@�C���X�g���[���쐬
	CComPtr<IStream> pStream;
	if(FAILED(SHCreateStreamOnFile(xml, STGM_CREATE | STGM_WRITE, &pStream))){
		return -1;
	}

	if(FAILED(pWriter->SetOutput(pStream))){
		return -1;
	}

	// �C���f���g�L����
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
// �v�f�̏�������
//=========================================
/** �v�f���J�n */
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

/** �v�f���I�� */
LONG XMLWriter::EndElement()
{
	// �������I��
	if(FAILED(pWriter->WriteFullEndElement()))
		return -1;
	return 0;
}


/** �v�f�ɕ�������������� */
LONG XMLWriter::WriteElement(const std::string&  name, const std::string& value)
{
	return WriteElement(Utility::ShiftjisToUnicode(name), Utility::ShiftjisToUnicode(value));
}
LONG XMLWriter::WriteElement(const std::wstring&  name, const std::wstring& value)
{
	// �v�f�̊J�n
	if(StartElement(name) != 0)
		return -1;

	// �l�̏�������
	if(pWriter->WriteString(value.c_str()) != 0)
		return -1;

	// �v�f�̏I��
	if(EndElement() != 0)
		return -1;

	return 0;
}

/** �v�f�ɕ�������������� */
LONG XMLWriter::WriteElement(const std::string&  name, const char value[])
{
	return WriteElement(name, (std::string)value);
}
LONG XMLWriter::WriteElement(const std::wstring&  name, const WCHAR value[])
{
	return WriteElement(name, (std::wstring)value);
}

/** �v�f�ɐ���(32bit)���������� */
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
/** �v�f�ɐ���(64bit)���������� */
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

/** �v�f�ɕ��������������������� */
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

/** �v�f�Ɏ������������� */
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

/** �v�f�ɘ_���l���������� */
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

/** �v�f��GUID���������� */
LONG XMLWriter::WriteElement(const std::string&  name, const boost::uuids::uuid& value)
{
	return this->WriteElement(Utility::ShiftjisToUnicode(name), value);
}
LONG XMLWriter::WriteElement(const std::wstring&  name, const boost::uuids::uuid& value)
{
	return this->WriteElement(name, boost::lexical_cast<std::wstring>(value));
}


//=======================================
// �v�f�ɕ������ǉ�
//=======================================
/** �v�f�ɕ�������������� */
LONG XMLWriter::AddElementString(const std::string& value)
{
	return this->AddElementString(Utility::ShiftjisToUnicode(value));
}
LONG XMLWriter::AddElementString(const std::wstring& value)
{
	// �l�̏�������
	if(pWriter->WriteString(value.c_str()) != 0)
		return -1;
	return 0;
}
/** �v�f�ɕ�������������� */
LONG XMLWriter::AddElementString(const char value[])
{
	return this->AddElementString((std::string)value);
}
LONG XMLWriter::AddElementString(const WCHAR value[])
{
	return this->AddElementString((std::wstring)value);
}
/** �v�f�ɐ������������� */
LONG XMLWriter::AddElementString(__int32 value)
{
	WCHAR szBuf[256];
	swprintf(szBuf, L"%d", value);
	return this->AddElementString((std::wstring)szBuf);
}
/** �v�f�ɐ���(63bit)���������� */
LONG XMLWriter::AddElementString(__int64 value)
{
	WCHAR szBuf[256];
	swprintf(szBuf, L"%lld", value);
	return this->AddElementString((std::wstring)szBuf);
}
/** �v�f�ɕ��������������������� */
LONG XMLWriter::AddElementString(unsigned long value)
{
	WCHAR szBuf[256];
	swprintf(szBuf, L"%ul", value);
	return this->AddElementString((std::wstring)szBuf);
}
/** �v�f�Ɏ������������� */
LONG XMLWriter::AddElementString(double value)
{
	WCHAR szBuf[256];
	swprintf(szBuf, L"%f", value);
	return this->AddElementString((std::wstring)szBuf);
}
/** �v�f�ɘ_���l���������� */
LONG XMLWriter::AddElementString(bool value)
{
	if(value)
		return this->AddElementString(L"true");
	else
		return this->AddElementString(L"false");
}
/** �v�f��GUID���������� */
LONG XMLWriter::AddElementString(const boost::uuids::uuid& value)
{
	return this->AddElementString(boost::lexical_cast<std::wstring>(value));
}



//=======================================
// �v�f�ɑ�����ǉ�
//=======================================
/** �v�f�ɑ�����ǉ� */
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

/** �v�f�ɑ�����ǉ�(������) */
LONG XMLWriter::AddAttribute(const std::string& id, const char value[])
{
	return AddAttribute(Utility::ShiftjisToUnicode(id), Utility::ShiftjisToUnicode(value));
}
LONG XMLWriter::AddAttribute(const std::wstring& id, const WCHAR value[])
{
	return AddAttribute(id, (std::wstring)value);
}

/** �v�f�ɑ�����ǉ�(����32bit) */
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

/** �v�f�ɑ�����ǉ�(����64bit) */
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

/** �v�f�ɑ�����ǉ�(����) */
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

/** �v�f�ɑ�����ǉ�(�_���l) */
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

/** �v�f�ɑ�����ǉ�����(GUID) */
LONG XMLWriter::AddAttribute(const std::string& id, const boost::uuids::uuid& value)
{
	return this->AddAttribute(Utility::ShiftjisToUnicode(id), value);
}
LONG XMLWriter::AddAttribute(const std::wstring& id, const boost::uuids::uuid& value)
{
	return this->AddAttribute(id, boost::lexical_cast<std::wstring>(value));
}