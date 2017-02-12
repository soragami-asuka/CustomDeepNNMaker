//==================================
// XML�ǂݏ����p�N���X
//==================================
#pragma once

#ifdef XMLUTILITY_EXPORTS
#define XMLUTILITY_API __declspec(dllexport)
#else
#define XMLUTILITY_API __declspec(dllimport)
#endif

#include<string>
#include<vector>
#include<map>

#include<boost/shared_ptr.hpp>
#include<./boost/uuid/uuid.hpp>

namespace XMLUtility
{
	enum NodeType
	{
		XmlNodeType_None	= 0,
		XmlNodeType_Element	= 1,
		XmlNodeType_Attribute	= 2,
		XmlNodeType_Text	= 3,
		XmlNodeType_CDATA	= 4,
		XmlNodeType_ProcessingInstruction	= 7,
		XmlNodeType_Comment	= 8,
		XmlNodeType_DocumentType	= 10,
		XmlNodeType_Whitespace	= 13,
		XmlNodeType_EndElement	= 15,
		XmlNodeType_XmlDeclaration	= 17,
		_XmlNodeType_Last	= 17
	};


	class XMLUTILITY_API IXmlWriter
	{
	public:
		/** �R���X�g���N�^ */
		IXmlWriter(){}
		/** �f�X�g���N�^ */
		virtual ~IXmlWriter(){}

	public:
		/** �v�f���J�n */
		virtual LONG StartElement(const std::string&  name) = 0;
		virtual LONG StartElement(const std::wstring& name) = 0;

		/** �v�f���I�� */
		virtual LONG EndElement() = 0;


		/** �v�f�ɕ�������������� */
		virtual LONG WriteElement(const std::string&  name, const std::string& value) = 0;
		virtual LONG WriteElement(const std::wstring&  name, const std::wstring& value) = 0;
		/** �v�f�ɕ�������������� */
		virtual LONG WriteElement(const std::string&  name, const char value[]) = 0;
		virtual LONG WriteElement(const std::wstring&  name, const WCHAR value[]) = 0;
		/** �v�f�ɐ���(32bit)���������� */
		virtual LONG WriteElement(const std::string&  name, __int32 value) = 0;
		virtual LONG WriteElement(const std::wstring&  name, __int32 value) = 0;
		/** �v�f�ɐ���(64bit)���������� */
		virtual LONG WriteElement(const std::string&  name, __int64 value) = 0;
		virtual LONG WriteElement(const std::wstring&  name, __int64 value) = 0;
		/** �v�f�ɕ��������������������� */
		virtual LONG WriteElement(const std::string&  name, unsigned long value) = 0;
		virtual LONG WriteElement(const std::wstring&  name, unsigned long value) = 0;
		/** �v�f�Ɏ������������� */
		virtual LONG WriteElement(const std::string&  name, double value) = 0;
		virtual LONG WriteElement(const std::wstring&  name, double value) = 0;
		/** �v�f�ɘ_���l���������� */
		virtual LONG WriteElement(const std::string&  name, bool value) = 0;
		virtual LONG WriteElement(const std::wstring&  name, bool value) = 0;
		/** �v�f��GUID���������� */
		virtual LONG WriteElement(const std::string&  name, const boost::uuids::uuid& value) = 0;
		virtual LONG WriteElement(const std::wstring&  name, const boost::uuids::uuid& value) = 0;


		/** �v�f�ɕ�������������� */
		virtual LONG AddElementString(const std::string& value) = 0;
		virtual LONG AddElementString(const std::wstring& value) = 0;
		/** �v�f�ɕ�������������� */
		virtual LONG AddElementString(const char value[]) = 0;
		virtual LONG AddElementString(const WCHAR value[]) = 0;
		/** �v�f�ɐ������������� */
		virtual LONG AddElementString(int value) = 0;
		/** �v�f�ɕ��������������������� */
		virtual LONG AddElementString(unsigned long value) = 0;
		/** �v�f�Ɏ������������� */
		virtual LONG AddElementString(double value) = 0;
		/** �v�f�ɘ_���l���������� */
		virtual LONG AddElementString(bool value) = 0;
		/** �v�f��GUID���������� */
		virtual LONG AddElementString(const boost::uuids::uuid& value) = 0;


		/** �v�f�ɑ�����ǉ� */
		virtual LONG AddAttribute(const std::string& id, const std::string& value) = 0;
		virtual LONG AddAttribute(const std::wstring& id, const std::wstring& value) = 0;
		/** �v�f�ɑ�����ǉ�(������) */
		virtual LONG AddAttribute(const std::string& id, const char value[]) = 0;
		virtual LONG AddAttribute(const std::wstring& id, const WCHAR value[]) = 0;
		/** �v�f�ɑ�����ǉ�(����32bit) */
		virtual LONG AddAttribute(const std::string& id, __int32 value) = 0;
		virtual LONG AddAttribute(const std::wstring& id, __int32 value) = 0;
		/** �v�f�ɑ�����ǉ�(����64bit) */
		virtual LONG AddAttribute(const std::string& id, __int64 value) = 0;
		virtual LONG AddAttribute(const std::wstring& id, __int64 value) = 0;
		/** �v�f�ɑ�����ǉ�(�{���x����) */
		virtual LONG AddAttribute(const std::string& id, double value) = 0;
		virtual LONG AddAttribute(const std::wstring& id, double value) = 0;
		/** �v�f�ɑ�����ǉ�(�_���l) */
		virtual LONG AddAttribute(const std::string& id, bool value) = 0;
		virtual LONG AddAttribute(const std::wstring& id, bool value) = 0;
		/** �v�f�ɑ�����ǉ�����(GUID) */
		virtual LONG AddAttribute(const std::string& strAttrName, const boost::uuids::uuid& value) = 0;
		virtual LONG AddAttribute(const std::wstring& strAttrName, const boost::uuids::uuid& value) = 0;
	};
	typedef boost::shared_ptr<IXmlWriter> XMLWriterPtr;

	class XMLUTILITY_API IXmlReader
	{
	public:
		/** �R���X�g���N�^ */
		IXmlReader(){}
		/** �f�X�g���N�^ */
		virtual ~IXmlReader(){}

	public:
		/** ���̃m�[�h��ǂݍ��� */
		virtual LONG Read(NodeType& nodeType) = 0;

		/** ���݂̗v�f�����擾���� */
		virtual std::string GetElementName() = 0;
		virtual std::wstring GetElementNameW() = 0;


		/** �l�𕶎���Ŏ擾���� */
		virtual std::string ReadElementValueString() = 0;
		virtual std::wstring ReadElementValueWString() = 0;

		/** �l��10�i�����Ŏ擾���� */
		virtual __int32 ReadElementValueInt32() = 0;
		virtual __int64 ReadElementValueInt64() = 0;
		/** �l��16�i�����Ŏ擾���� */
		virtual __int32 ReadElementValueIntX32() = 0;
		virtual __int64 ReadElementValueIntX64() = 0;
		/** �l�������Ŏ擾���� */
		virtual double ReadElementValueDouble() = 0;
		/** �l��_���l�Ŏ擾���� */
		virtual bool ReadElementValueBool() = 0;
		/** �l��GUID�Ŏ擾���� */
		virtual boost::uuids::uuid ReadElementValueGUID() = 0;


		/** �l�𕶎���z��̔ԍ��Ŏ擾���� */
		virtual LONG ReadElementValueEnum(const std::string lpName[], LONG valueCount) = 0;
		virtual LONG ReadElementValueEnum(const std::wstring lpName[], LONG valueCount) = 0;

		/** �l�𕶎���z��̔ԍ��Ŏ擾���� */
		virtual LONG ReadElementValueEnum(const std::vector<std::string>& lpName) = 0;
		virtual LONG ReadElementValueEnum(const std::vector<std::wstring>& lpName) = 0;

		/** ���������X�g�Ŏ擾���� */
		virtual std::map<std::string, std::string> ReadAttributeList() = 0;
		virtual std::map<std::wstring, std::wstring> ReadAttributeListW() = 0;
	};
	typedef boost::shared_ptr<IXmlReader> XMLReaderPtr;


	//==========================================
	// �֐���`
	//==========================================
	/** XML�������݃N���X���쐬���� */
	XMLUTILITY_API XMLWriterPtr CreateXMLWriter(const std::string& filePath);
	XMLUTILITY_API XMLWriterPtr CreateXMLWriter(const std::wstring& filePath);

	/** XML�ǂݍ��݃N���X���쐬���� */
	XMLUTILITY_API XMLReaderPtr CreateXMLReader(const std::string& filePath);
	XMLUTILITY_API XMLReaderPtr CreateXMLReader(const std::wstring& filePath);
}

