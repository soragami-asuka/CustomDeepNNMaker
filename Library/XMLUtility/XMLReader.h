//==================================
// XML�ǂݍ��ݗp�N���X
//==================================
#pragma once

#include<Windows.h>
#include"XMLUtility.h"

#include<atlbase.h>  // CComPtr���g�p���邽��
#include<xmllite.h>

namespace XMLUtility
{
	class XMLReader : public XMLUtility::IXmlReader
	{
	private:
		CComPtr<::IXmlReader> pReader;

	public:
		/** �R���X�g���N�^ */
		XMLReader();
		/** �f�X�g���N�^ */
		virtual ~XMLReader();

	public:
		/** ������ */
		LONG Initialize(const std::wstring& filePath);

	public:
		/** ���̃m�[�h��ǂݍ��� */
		LONG Read(NodeType& nodeType);

		/** ���݂̗v�f�����擾���� */
		std::string GetElementName();
		std::wstring GetElementNameW();


		/** �l�𕶎���Ŏ擾���� */
		std::string ReadElementValueString();
		std::wstring ReadElementValueWString();

		/** �l��10�i�����Ŏ擾���� */
		__int32 ReadElementValueInt32();
		__int64 ReadElementValueInt64();
		/** �l��16�i�����Ŏ擾���� */
		__int32 ReadElementValueIntX32();
		__int64 ReadElementValueIntX64();
		/** �l�������Ŏ擾���� */
		double ReadElementValueDouble();
		/** �l��_���l�Ŏ擾���� */
		bool ReadElementValueBool();
		/** �l��GUID�Ŏ擾���� */
		boost::uuids::uuid ReadElementValueGUID();


		/** �l�𕶎���z��̔ԍ��Ŏ擾���� */
		LONG ReadElementValueEnum(const std::string lpName[], LONG valueCount);
		LONG ReadElementValueEnum(const std::wstring lpName[], LONG valueCount);

		/** �l�𕶎���z��̔ԍ��Ŏ擾���� */
		LONG ReadElementValueEnum(const std::vector<std::string>& lpName);
		LONG ReadElementValueEnum(const std::vector<std::wstring>& lpName);

		/** ���������X�g�Ŏ擾���� */
		std::map<std::string, std::string> ReadAttributeList();
		std::map<std::wstring, std::wstring> ReadAttributeListW();
	};
}