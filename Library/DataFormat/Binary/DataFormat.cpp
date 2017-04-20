// StringArray.cpp : DLL �A�v���P�[�V�����p�ɃG�N�X�|�[�g�����֐����`���܂��B
//

#include "stdafx.h"


#include"DataFormat.h"

#include<string>
#include<vector>
#include<list>
#include<set>
#include<map>

#include<boost/property_tree/ptree.hpp>
#include<boost/property_tree/xml_parser.hpp>
#include <boost/iostreams/device/file_descriptor.hpp>
#include <boost/iostreams/stream.hpp>

#include"Library/Common/StringUtility/StringUtility.h"

#include"DataFormatItem.h"

namespace Gravisbell {
namespace DataFormat {
namespace Binary {


	/** ������̔z���ǂݍ��ރf�[�^�t�H�[�}�b�g���쐬���� */
	GRAVISBELL_LIBRAY_API IDataFormat* CreateDataFormat(const wchar_t i_szName[], const wchar_t i_szText[])
	{
		return new CDataFormat(i_szName, i_szText);
	}
	/** ������̔z���ǂݍ��ރf�[�^�t�H�[�}�b�g���쐬���� */
	extern GRAVISBELL_LIBRAY_API IDataFormat* CreateDataFormatFromXML(const wchar_t szXMLFilePath[])
	{
		using namespace StringUtility;

		// XML�t�@�C���̓ǂݍ���
		boost::property_tree::ptree pXmlTree;
		boost::iostreams::file_descriptor_source fs(UnicodeToShiftjis(szXMLFilePath));
		boost::iostreams::stream<boost::iostreams::file_descriptor_source> fsstream(fs);
		try
		{
			boost::property_tree::read_xml(fsstream, pXmlTree);
		}
		catch(boost::exception& e)
		{
			e;
			return NULL;
		}

		CDataFormat* pDataFormat = NULL;
		try
		{
			// ���O
			std::wstring name;
			if(boost::optional<std::string> pValue = pXmlTree.get_optional<std::string>("DataFormat.Name"))
			{
				name = UTF8toUnicode(pValue.get());
			}
			// �e�L�X�g
			std::wstring text;
			if(boost::optional<std::string> pValue = pXmlTree.get_optional<std::string>("DataFormat.Text"))
			{
				text = UTF8toUnicode(pValue.get());
			}

			// bool�l�̒l
			std::map<std::wstring, BoolValue>	lpBoolValue;	/**< bool�l��F32�ɕϊ�����ݒ�l�̈ꗗ.	<�f�[�^��ʖ�, �ϊ��f�[�^> */
			lpBoolValue[L""] = BoolValue();
			for(const boost::property_tree::ptree::value_type &it : pXmlTree.get_child("DataFormat.BoolValue"))
			{
				if(it.first == "true" || it.first == "false")
				{
					// ��������J�e�S�����擾
					std::wstring category = L"";
					if(boost::optional<std::string> pValue = it.second.get_optional<std::string>("<xmlattr>.category"))
					{
						category = UTF8toUnicode(pValue.get());
					}

					if(it.first == "true")
						lpBoolValue[category].trueValue = (F32)atof(it.second.data().c_str());
					else if(it.first == "false")
						lpBoolValue[category].falseValue = (F32)atof(it.second.data().c_str());
				}
			}


			// �f�[�^�t�H�[�}�b�g���쐬
			pDataFormat = new CDataFormat();


			// Channel�̓ǂݍ���
			for(const boost::property_tree::ptree::value_type &it : pXmlTree.get_child("DataFormat.Channel"))
			{
				// id�̎擾
				std::wstring id = L"";
				if(boost::optional<std::string> pValue = it.second.get_optional<std::string>("<xmlattr>.id"))
				{
					id = UTF8toUnicode(pValue.get());
				}
				// category�̎擾
				std::wstring category = L"";
				if(boost::optional<std::string> pValue = it.second.get_optional<std::string>("<xmlattr>.category"))
				{
					category = UTF8toUnicode(pValue.get());
				}

				// bool�^�̒l���擾
				BoolValue boolValue = lpBoolValue[L""];
				if(lpBoolValue.count(category))
					boolValue = lpBoolValue[category];

				if(it.first == "String")
				{
					enum UseType
					{
						USETYPE_BIT,
						USETYPE_BITARRAY,
						USETYPE_BITARRAY_ENUM
					};
					UseType useType = USETYPE_BITARRAY;

					// �g�p���@���擾
					if(boost::optional<std::string> pValue = it.second.get_optional<std::string>("<xmlattr>.useType"))
					{
						if(pValue.get() == "bit")
							useType = USETYPE_BIT;
						else if(pValue.get() == "bit_array")
							useType = USETYPE_BITARRAY;
						else if(pValue.get() == "bit_array_enum")
							useType = USETYPE_BITARRAY_ENUM;
					}

					switch(useType)
					{
					case USETYPE_BIT:
						{
							// false�l���
							std::list<std::wstring> lpFalseString;
							std::vector<const wchar_t*> lpFalseStringPointer;
							if(auto& pTreeEnum = it.second.get_child_optional("false"))
							{
								for(const boost::property_tree::ptree::value_type &it_enum : pTreeEnum.get().get_child(""))
								{
									if(it_enum.first == "item")
									{
										lpFalseString.push_back(UTF8toUnicode(it_enum.second.data()));
										lpFalseStringPointer.push_back(lpFalseString.rbegin()->c_str());
									}
								}
							}
							// True�l���
							std::list<std::wstring> lpTrueString;
							std::vector<const wchar_t*> lpTrueStringPointer;
							if(auto& pTreeEnum = it.second.get_child_optional("true"))
							{
								for(const boost::property_tree::ptree::value_type &it_enum : pTreeEnum.get().get_child(""))
								{
									if(it_enum.first == "item")
									{
										lpTrueString.push_back(UTF8toUnicode(it_enum.second.data()));
										lpTrueStringPointer.push_back(lpTrueString.rbegin()->c_str());
									}
								}
							}

							pDataFormat->AddDataFormatStringToBit(
								id.c_str(), category.c_str(),
								lpFalseStringPointer.size(), &lpFalseStringPointer[0], lpTrueStringPointer.size(), &lpTrueStringPointer[0],
								boolValue.falseValue, boolValue.trueValue);
						}
						break;

					case USETYPE_BITARRAY:
					default:
						{
							// �t�H�[�}�b�g��ǉ�
							pDataFormat->AddDataFormatStringToBitArray(id.c_str(), category.c_str(), boolValue.falseValue, boolValue.trueValue); 
						}
						break;

					case USETYPE_BITARRAY_ENUM:
						{
							// enum�l���
							std::list<std::wstring> lpEnumString;
							std::vector<const wchar_t*> lpEnumStringPointer;
							std::wstring defaultString = L"";
							if(auto& pTreeEnum = it.second.get_child_optional("Enum"))
							{
								for(const boost::property_tree::ptree::value_type &it_enum : pTreeEnum.get().get_child(""))
								{
									if(it_enum.first == "item")
									{
										lpEnumString.push_back(UTF8toUnicode(it_enum.second.data()));
										lpEnumStringPointer.push_back(lpEnumString.rbegin()->c_str());

										if(boost::optional<std::string> pValue = it_enum.second.get_optional<std::string>("<xmlattr>.default"))
										{
											if(pValue.get() == "true")
												defaultString = UTF8toUnicode(it_enum.second.data());
										}
									}
								}
							}

							// �t�H�[�}�b�g��ǉ�
							pDataFormat->AddDataFormatStringToBitArrayEnum(id.c_str(), category.c_str(), lpEnumStringPointer.size(), &lpEnumStringPointer[0], defaultString.c_str(), boolValue.falseValue, boolValue.trueValue); 
						}
						break;
					}
				}
				else if(it.first == "Float")
				{
					enum NormalizeType
					{
						NORMALIZETYPE_NONE,		// ���K�����Ȃ�
						NORMALIZETYPE_MINMAX,	// �S�f�[�^�̍ŏ��l�A�ő�l�����ɐ��K������
						NORMALIZETYPE_VALUE,	// �ŏ��l�A�ő�l���w�肵�Đ��K������
						NORMALIZETYPE_SDEV,		// �S�f�[�^�̕��ϒl�A�W���΍������ɐ��K������
					};
					NormalizeType normalizeType = NORMALIZETYPE_NONE;

					// ���K����ʂ��擾
					if(boost::optional<std::string> pValue = it.second.get_optional<std::string>("<xmlattr>.normalize"))
					{
						if(pValue.get() == "none")
							normalizeType = NORMALIZETYPE_NONE;
						else if(pValue.get() == "min_max")
							normalizeType = NORMALIZETYPE_MINMAX;
						else if(pValue.get() == "value")
							normalizeType = NORMALIZETYPE_VALUE;
						else if(pValue.get() == "average_deviation")
							normalizeType = NORMALIZETYPE_SDEV;
					}

					// �ݒ�ŏ��l, �ő�l���擾����
					F32 minValue = 0.0f;
					F32 maxValue = 1.0f;
					if(boost::optional<float> pValue = it.second.get_optional<float>("min"))
						minValue = pValue.get();
					if(boost::optional<float> pValue = it.second.get_optional<float>("max"))
						maxValue = pValue.get();

					// �o�͍ŏ��l�A�ő�l���擾����
					if(boost::optional<float> pValue = it.second.get_optional<float>("output_min"))
						boolValue.falseValue = pValue.get();
					if(boost::optional<float> pValue = it.second.get_optional<float>("output_max"))
						boolValue.trueValue = pValue.get();

					switch(normalizeType)
					{
					case NORMALIZETYPE_NONE:		// ���K�����Ȃ�
					default:
						pDataFormat->AddDataFormatFloat(id.c_str(), category.c_str());
						break;
					case NORMALIZETYPE_MINMAX:	// �S�f�[�^�̍ŏ��l�A�ő�l�����ɐ��K������
						pDataFormat->AddDataFormatFloatNormalizeMinMax(id.c_str(), category.c_str(), boolValue.falseValue, boolValue.trueValue);
						break;
					case NORMALIZETYPE_VALUE:	// �ŏ��l�A�ő�l���w�肵�Đ��K������
						pDataFormat->AddDataFormatFloatNormalizeValue(id.c_str(), category.c_str(), minValue, maxValue, boolValue.falseValue, boolValue.trueValue);
						break;
					case NORMALIZETYPE_SDEV:		// �S�f�[�^�̕��ϒl�A�W���΍������ɐ��K������
						pDataFormat->AddDataFormatFloatNormalizeAverageDeviation(id.c_str(), category.c_str(), minValue, maxValue, boolValue.falseValue, boolValue.trueValue);
						break;
					}
				}
			}
		}
		catch(boost::exception& e)
		{
			e;
			if(pDataFormat)
				delete pDataFormat;
			return NULL;
		}

		return pDataFormat;
	}


}	// Binary
}	// DataFormat
}	// Gravisbell


