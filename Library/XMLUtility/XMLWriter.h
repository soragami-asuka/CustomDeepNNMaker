//==================================
// XML�������ݗp�N���X
//==================================
#pragma once

#include<Windows.h>
#include"XMLUtility.h"

#include<atlbase.h>  // CComPtr���g�p���邽��
#include<xmllite.h>

namespace XMLUtility
{
	class XMLWriter : public IXmlWriter
	{
	private:
		CComPtr<::IXmlWriter> pWriter;

	public:
		/** �R���X�g���N�^ */
		XMLWriter();
		/** �f�X�g���N�^ */
		~XMLWriter();

	public:
		/** ������ */
		LONG Initialize(const std::wstring& filePath);

	public:
		/** �v�f���J�n */
		LONG StartElement(const std::string&  name);
		LONG StartElement(const std::wstring& name);

		/** �v�f���I�� */
		LONG EndElement();


	public:
		/** �v�f�ɕ�������������� */
		LONG WriteElement(const std::string&  name, const std::string& value);
		LONG WriteElement(const std::wstring&  name, const std::wstring& value);
		/** �v�f�ɕ�������������� */
		LONG WriteElement(const std::string&  name, const char value[]);
		LONG WriteElement(const std::wstring&  name, const WCHAR value[]);
		/** �v�f�ɐ���(32bit)���������� */
		LONG WriteElement(const std::string&  name, __int32 value);
		LONG WriteElement(const std::wstring&  name, __int32 value);
		/** �v�f�ɐ���(64bit)���������� */
		LONG WriteElement(const std::string&  name, __int64 value);
		LONG WriteElement(const std::wstring&  name, __int64 value);
		/** �v�f�ɕ��������������������� */
		LONG WriteElement(const std::string&  name, unsigned long value);
		LONG WriteElement(const std::wstring&  name, unsigned long value);
		/** �v�f�Ɏ������������� */
		LONG WriteElement(const std::string&  name, double value);
		LONG WriteElement(const std::wstring&  name, double value);
		/** �v�f�ɘ_���l���������� */
		LONG WriteElement(const std::string&  name, bool value);
		LONG WriteElement(const std::wstring&  name, bool value);
		/** �v�f��GUID���������� */
		LONG WriteElement(const std::string&  name, const boost::uuids::uuid& value);
		LONG WriteElement(const std::wstring&  name, const boost::uuids::uuid& value);


		/** �v�f�ɕ�������������� */
		LONG AddElementString(const std::string& value);
		LONG AddElementString(const std::wstring& value);
		/** �v�f�ɕ�������������� */
		LONG AddElementString(const char value[]);
		LONG AddElementString(const WCHAR value[]);
		/** �v�f�ɐ���(32bit)���������� */
		LONG AddElementString(__int32 value);
		/** �v�f�ɐ���(63bit)���������� */
		LONG AddElementString(__int64 value);
		/** �v�f�ɕ��������������������� */
		LONG AddElementString(unsigned long value);
		/** �v�f�Ɏ������������� */
		LONG AddElementString(double value);
		/** �v�f�ɘ_���l���������� */
		LONG AddElementString(bool value);
		/** �v�f��GUID���������� */
		LONG AddElementString(const boost::uuids::uuid& value);


		/** �v�f�ɑ�����ǉ� */
		LONG AddAttribute(const std::string& id, const std::string& value);
		LONG AddAttribute(const std::wstring& id, const std::wstring& value);
		/** �v�f�ɑ�����ǉ�(������) */
		LONG AddAttribute(const std::string& id, const char value[]);
		LONG AddAttribute(const std::wstring& id, const WCHAR value[]);
		/** �v�f�ɑ�����ǉ�(����32bit) */
		LONG AddAttribute(const std::string& id, __int32 value);
		LONG AddAttribute(const std::wstring& id, __int32 value);
		/** �v�f�ɑ�����ǉ�(����64bit) */
		LONG AddAttribute(const std::string& id, __int64 value);
		LONG AddAttribute(const std::wstring& id, __int64 value);
		/** �v�f�ɑ�����ǉ�(�{���x����) */
		LONG AddAttribute(const std::string& id, double value);
		LONG AddAttribute(const std::wstring& id, double value);
		/** �v�f�ɑ�����ǉ�(�_���l) */
		LONG AddAttribute(const std::string& id, bool value);
		LONG AddAttribute(const std::wstring& id, bool value);
		/** �v�f�ɑ�����ǉ�����(GUID) */
		LONG AddAttribute(const std::string& id, const boost::uuids::uuid& value);
		LONG AddAttribute(const std::wstring& id, const boost::uuids::uuid& value);
	};
}
