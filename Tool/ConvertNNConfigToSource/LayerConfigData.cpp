//======================================
// ���C���[�̐ݒ���ɂ��ċL��
//======================================
#include"stdafx.h"

#include<boost/property_tree/ptree.hpp>
#include<boost/property_tree/xml_parser.hpp>
#include<boost/foreach.hpp>
#include<boost/lexical_cast.hpp>
#include<boost/regex.hpp>

#include"LayerConfigData.h"

#include"ConfigString.h"

using namespace CustomDeepNNLibrary;

namespace
{
	INNLayerConfigEx* ReadLayerConfigXML(boost::property_tree::ptree& pTree, const GUID& layerCode, const VersionCode& versionCode)
	{
		INNLayerConfigEx* pConfig = CustomDeepNNLibrary::CreateEmptyLayerConfig(layerCode, versionCode);
		if(pConfig == NULL)
			return NULL;

		// �����̃A�C�e����ǂݍ���
		for(const boost::property_tree::ptree::value_type &it : pTree.get_child(""))
		{
			if(it.first == "Int")
			{
				std::wstring id;
				std::wstring name;
				std::wstring text;
				int minValue = 0;
				int maxValue = 0;
				int defaultValue = 0;

				// ID
				if(boost::optional<std::string> pValue = it.second.get_optional<std::string>("<xmlattr>.id"))
				{
					id = UTF8toUnicode(pValue.get());
				}
				else
				{
					delete pConfig;
					return NULL;
				}
				// ���O
				if(boost::optional<std::string> pValue = it.second.get_optional<std::string>("Name"))
				{
					name = UTF8toUnicode(pValue.get());
				}
				// �e�L�X�g
				if(boost::optional<std::string> pValue = it.second.get_optional<std::string>("Text"))
				{
					text = UTF8toUnicode(pValue.get());
				}
				// �ŏ��l
				if(boost::optional<int> pValue = it.second.get_optional<int>("Min"))
				{
					minValue = pValue.get();
				}
				// �ő�l
				if(boost::optional<int> pValue = it.second.get_optional<int>("Max"))
				{
					maxValue = pValue.get();
				}
				// �f�t�H���g�l
				if(boost::optional<int> pValue = it.second.get_optional<int>("Default"))
				{
					defaultValue = pValue.get();
				}

				// �A�C�e�����쐬����
				INNLayerConfigItem_Int* pItem = CreateLayerCofigItem_Int(id.c_str(), name.c_str(), text.c_str(), minValue, maxValue, defaultValue);
				if(pItem == NULL)
				{
					delete pConfig;
					return NULL;
				}

				pConfig->AddItem(pItem);
			}
			else if(it.first == "Float")
			{
				std::wstring id;
				std::wstring name;
				std::wstring text;
				float minValue = 0;
				float maxValue = 0;
				float defaultValue = 0;

				// ID
				if(boost::optional<std::string> pValue = it.second.get_optional<std::string>("<xmlattr>.id"))
				{
					id = UTF8toUnicode(pValue.get());
				}
				else
				{
					delete pConfig;
					return NULL;
				}
				// ���O
				if(boost::optional<std::string> pValue = it.second.get_optional<std::string>("Name"))
				{
					name = UTF8toUnicode(pValue.get());
				}
				// �e�L�X�g
				if(boost::optional<std::string> pValue = it.second.get_optional<std::string>("Text"))
				{
					text = UTF8toUnicode(pValue.get());
				}
				// �ŏ��l
				if(boost::optional<float> pValue = it.second.get_optional<float>("Min"))
				{
					minValue = pValue.get();
				}
				// �ő�l
				if(boost::optional<float> pValue = it.second.get_optional<float>("Max"))
				{
					maxValue = pValue.get();
				}
				// �f�t�H���g�l
				if(boost::optional<float> pValue = it.second.get_optional<float>("Default"))
				{
					defaultValue = pValue.get();
				}

				// �A�C�e�����쐬����
				INNLayerConfigItem_Float* pItem = CreateLayerCofigItem_Float(id.c_str(), name.c_str(), text.c_str(), minValue, maxValue, defaultValue);
				if(pItem == NULL)
				{
					delete pConfig;
					return NULL;
				}

				pConfig->AddItem(pItem);
			}
			else if(it.first == "Bool")
			{
				std::wstring id;
				std::wstring name;
				std::wstring text;
				bool defaultValue = 0;

				// ID
				if(boost::optional<std::string> pValue = it.second.get_optional<std::string>("<xmlattr>.id"))
				{
					id = UTF8toUnicode(pValue.get());
				}
				else
				{
					delete pConfig;
					return NULL;
				}
				// ���O
				if(boost::optional<std::string> pValue = it.second.get_optional<std::string>("Name"))
				{
					name = UTF8toUnicode(pValue.get());
				}
				// �e�L�X�g
				if(boost::optional<std::string> pValue = it.second.get_optional<std::string>("Text"))
				{
					text = UTF8toUnicode(pValue.get());
				}
				// �f�t�H���g�l
				if(boost::optional<bool> pValue = it.second.get_optional<bool>("Default"))
				{
					defaultValue = pValue.get();
				}

				// �A�C�e�����쐬����
				INNLayerConfigItem_Bool* pItem = CreateLayerCofigItem_Bool(id.c_str(), name.c_str(), text.c_str(), defaultValue);
				if(pItem == NULL)
				{
					delete pConfig;
					return NULL;
				}

				pConfig->AddItem(pItem);
			}
			else if(it.first == "String")
			{
				std::wstring id;
				std::wstring name;
				std::wstring text;
				std::wstring defaultValue;

				// ID
				if(boost::optional<std::string> pValue = it.second.get_optional<std::string>("<xmlattr>.id"))
				{
					id = UTF8toUnicode(pValue.get());
				}
				else
				{
					delete pConfig;
					return NULL;
				}
				// ���O
				if(boost::optional<std::string> pValue = it.second.get_optional<std::string>("Name"))
				{
					name = UTF8toUnicode(pValue.get());
				}
				// �e�L�X�g
				if(boost::optional<std::string> pValue = it.second.get_optional<std::string>("Text"))
				{
					text = UTF8toUnicode(pValue.get());
				}
				// �f�t�H���g�l
				if(boost::optional<std::string> pValue = it.second.get_optional<std::string>("Default"))
				{
					defaultValue = UTF8toUnicode(pValue.get());
				}

				// �A�C�e�����쐬����
				INNLayerConfigItem_String* pItem = CreateLayerCofigItem_String(id.c_str(), name.c_str(), text.c_str(), defaultValue.c_str());
				if(pItem == NULL)
				{
					delete pConfig;
					return NULL;
				}

				pConfig->AddItem(pItem);
			}
			else if(it.first == "Enum")
			{
				std::wstring id;
				std::wstring name;
				std::wstring text;
				std::wstring defaultValue;

				// ID
				if(boost::optional<std::string> pValue = it.second.get_optional<std::string>("<xmlattr>.id"))
				{
					id = UTF8toUnicode(pValue.get());
				}
				else
				{
					delete pConfig;
					return NULL;
				}
				// ���O
				if(boost::optional<std::string> pValue = it.second.get_optional<std::string>("Name"))
				{
					name = UTF8toUnicode(pValue.get());
				}
				// �e�L�X�g
				if(boost::optional<std::string> pValue = it.second.get_optional<std::string>("Text"))
				{
					text = UTF8toUnicode(pValue.get());
				}
				// �f�t�H���g�l
				if(boost::optional<std::string> pValue = it.second.get_optional<std::string>("Default"))
				{
					defaultValue = UTF8toUnicode(pValue.get());
				}

				// �A�C�e�����쐬����
				INNLayerConfigItemEx_Enum* pItem = CreateLayerCofigItem_Enum(id.c_str(), name.c_str(), text.c_str());
				if(pItem == NULL)
				{
					delete pConfig;
					return NULL;
				}

				// �񋓒l��ǉ�����
				for(const boost::property_tree::ptree::value_type &it_item : it.second.get_child("Items"))
				{
					std::wstring item_id;
					std::wstring item_name;
					std::wstring item_text;

					// ID
					if(boost::optional<std::string> pValue = it_item.second.get_optional<std::string>("<xmlattr>.id"))
					{
						item_id = UTF8toUnicode(pValue.get());
					}
					else
					{
						delete pItem;
						delete pConfig;
						return NULL;
					}
					// ���O
					if(boost::optional<std::string> pValue = it_item.second.get_optional<std::string>("Name"))
					{
						item_name = UTF8toUnicode(pValue.get());
					}
					// �e�L�X�g
					if(boost::optional<std::string> pValue = it_item.second.get_optional<std::string>("Text"))
					{
						item_text = UTF8toUnicode(pValue.get());
					}

					pItem->AddEnumItem(item_id.c_str(), item_name.c_str(), item_text.c_str());
				}

				// �f�t�H���g�l��ݒ�
				pItem->SetDefaultItem(defaultValue.c_str());

				// �ݒ���ɒǉ�
				pConfig->AddItem(pItem);
			}
		}

		return pConfig;
	}

	/** ����������s�P�ʂŕ������Ĕz�� */
	int TextToStringArray(const std::wstring input, std::vector<std::wstring>& lpStringArray)
	{
		lpStringArray.clear();

		std::wstring buf = input;

		while(true)
		{
			std::size_t indexN = buf.find(L"\n");
			std::size_t indexRN = buf.find(L"\r\n");

			std::size_t index = std::min(indexN, indexRN);
			if(index == std::string::npos)
			{
				lpStringArray.push_back(buf);
				break;
			}

			std::size_t skipByte = 1;
			if(index == indexN)
				skipByte = 1;
			if(index == indexRN)
				skipByte = 2;

			lpStringArray.push_back(buf.substr(0, index));
			buf = buf.substr(index+skipByte);

			if(buf.size() == 0)
				break;
		}

		return 0;
	}
}


/** �R���X�g���N�^ */
LayerConfigData::LayerConfigData()
	:	guid	()					/**< ���ʃR�[�h */
	,	default_language	(L"")	/**< ��{���� */

	,	name	(L"")	/**< ���O */
	,	text	(L"")	/**< �����e�L�X�g */

	,	pStructure	(NULL)	/**< ���C���[�\����`��� */
	,	pLearn		(NULL)	/**< �w�K�ݒ��� */
{
}
/** �f�X�g���N�^ */
LayerConfigData::~LayerConfigData()
{
	if(this->pStructure)
		delete this->pStructure;
	if(this->pLearn)
		delete this->pLearn;
}

/** XML�t�@�C���������ǂݍ���.
	@param	configFilePath	�ǂݍ���XML�t�@�C���̃p�X
	@return	���������ꍇ0���Ԃ�. */
int LayerConfigData::ReadFromXMLFile(const boost::filesystem::wpath& configFilePath)
{
	// ����̃t�@�C����������
	if(this->pStructure)
		delete this->pStructure;
	if(this->pLearn)
		delete this->pLearn;


	// XML�t�@�C���̓ǂݍ���
	boost::property_tree::ptree pXmlTree;
	try
	{
		boost::property_tree::read_xml(configFilePath.generic_string(), pXmlTree);
	}
	catch(boost::exception& e)
	{
		e;
		return -1;
	}

	// ����ǂݍ���
	try
	{
		// GUID
		if(boost::optional<std::string> guid_str = pXmlTree.get_optional<std::string>("Config.<xmlattr>.guid"))
		{
			std::wstring buf = UTF8toUnicode(guid_str.get());

			// ������𕪉�
			boost::wregex reg(L"^([0-9a-fA-F]{8})-([0-9a-fA-F]{4})-([0-9a-fA-F]{4})-([0-9a-fA-F]{2})([0-9a-fA-F]{2})-([0-9a-fA-F]{2})([0-9a-fA-F]{2})([0-9a-fA-F]{2})([0-9a-fA-F]{2})([0-9a-fA-F]{2})([0-9a-fA-F]{2})$");
			boost::wsmatch match;
			if(boost::regex_search(buf, match, reg))
			{
				this->guid.Data1 = (unsigned long)wcstoul(match[1].str().c_str(), NULL, 16);
				this->guid.Data2 = (unsigned short)wcstoul(match[2].str().c_str(), NULL, 16);
				this->guid.Data3 = (unsigned short)wcstoul(match[3].str().c_str(), NULL, 16);
				for(int i=0; i<8; i++)
				{
					this->guid.Data4[i] = (unsigned char)wcstoul(match[4+i].str().c_str(), NULL, 16);
				}
			}
			else
			{
				return -1;
			}
		}
		else
		{
			return -1;
		}

		// �o�[�W����
		if(boost::optional<std::string> version_str = pXmlTree.get_optional<std::string>("Config.<xmlattr>.version"))
		{
			std::wstring buf = UTF8toUnicode(version_str.get());

			// ������𕪉�
			boost::wregex reg(L"^([0-9])\\.([0-9])\\.([0-9])\\.([0-9])$");
			boost::wsmatch match;
			if(boost::regex_search(buf, match, reg))
			{
				for(int i=0; i<4; i++)
				{
					this->version.lpData[i] = _wtoi(match[1+i].str().c_str());
				}
			}
			else
			{
				return -1;
			}
		}
		else
		{
			return -1;
		}

		// ��{����
		if(boost::optional<std::string> language_str = pXmlTree.get_optional<std::string>("Config.<xmlattr>.default-language"))
		{
			this->default_language = UTF8toUnicode(language_str.get());
		}
		else
		{
			this->default_language = L"ja";
		}

		// ���O
		if(boost::optional<std::string> name_str = pXmlTree.get_optional<std::string>("Config.Name"))
		{
			this->name = UTF8toUnicode(name_str.get());
		}
		else
		{
			this->name = L"";
		}
		
		// ������
		if(boost::optional<std::string> text_str = pXmlTree.get_optional<std::string>("Config.Text"))
		{
			this->text = UTF8toUnicode(text_str.get());
		}
		else
		{
			this->text = L"";
		}

		// ���C���[�\��
		if(auto &structure_tree = pXmlTree.get_child_optional("Config.Structure"))
		{
			if(this->pStructure)
				delete this->pStructure;

			this->pStructure = ::ReadLayerConfigXML(structure_tree.get(), this->guid, this->version);
			if(this->pStructure == NULL)
				return -1;
		}
		else
		{
			return -1;
		}

		// �w�K�ݒ�
		if(auto &learn_tree = pXmlTree.get_child_optional("Config.Learn"))
		{
			if(this->pLearn)
				delete this->pLearn;

			this->pLearn = ::ReadLayerConfigXML(learn_tree.get(), this->guid, this->version);
			if(this->pLearn == NULL)
				return -1;
		}
		else
		{
			return -1;
		}
	}
	catch(boost::exception& e)
	{
		e;
		return -1;
	}

	return 0;
}
/** C++����\�[�X�t�@�C���ɕϊ�/�o�͂���.
	.h/.cpp�t�@�C�������������.
	@param	exportDirPath	�o�͐�f�B���N�g���p�X
	@param	fileName		�o�̓t�@�C����.�g���q�͏���.
	@return ���������ꍇ0���Ԃ�. */
int LayerConfigData::ConvertToCPPFile(const boost::filesystem::wpath& exportDirPath, const std::wstring& fileName)const
{
	std::locale::global(std::locale("japanese"));

	boost::filesystem::wpath sourceFilePath = exportDirPath / (fileName + L".cpp");
	boost::filesystem::wpath headerFilePath = exportDirPath / (fileName + L".hpp");

	// GUID�𕶎���ɕϊ�
	std::wstring strGUID;
	{
		wchar_t szBuf[64];
		swprintf_s(szBuf, L"%08X-%04X-%04X-%02X%02X-%02X%02X%02X%02X%02X%02X",
			this->guid.Data1,
			this->guid.Data2,
			this->guid.Data3,
			this->guid.Data4[0], this->guid.Data4[1],
			this->guid.Data4[2], this->guid.Data4[3], this->guid.Data4[4], this->guid.Data4[5], this->guid.Data4[6], this->guid.Data4[7]);
		strGUID = szBuf;
	}
	
	// �w�b�_�t�@�C���̏o��
	{
		// �t�@�C���I�[�v��
		FILE* fp = _wfopen(headerFilePath.wstring().c_str(), L"w");
		if(fp == NULL)
			return -1;

		std::vector<std::wstring> lpText;
		TextToStringArray(this->text, lpText);

		fwprintf(fp, L"/*--------------------------------------------\n");
		fwprintf(fp, L" * FileName  : %ls\n", headerFilePath.filename().wstring().c_str());
		fwprintf(fp, L" * LayerName : %ls\n", this->name.c_str());
		fwprintf(fp, L" * guid      : %ls\n", strGUID.c_str());
		if(lpText.size() > 0)
		{
			fwprintf(fp, L" * \n");
			fwprintf(fp, L" * Text      : %ls\n", lpText[0].c_str());
			for(unsigned int i=1; i<lpText.size(); i++)
				fwprintf(fp, L" *           : %ls\n", lpText[i].c_str());
		}
		else
		{
		}
		fwprintf(fp, L"--------------------------------------------*/\n");
		fwprintf(fp, L"#ifndef __CUSTOM_DEEP_NN_LAYER_%s_H__\n", fileName.c_str());
		fwprintf(fp, L"#define __CUSTOM_DEEP_NN_LAYER_%s_H__\n", fileName.c_str());
		fwprintf(fp, L"\n");
		fwprintf(fp, L"#define EXPORT_API extern \"C\" __declspec(dllexport)\n");
		fwprintf(fp, L"\n");
		fwprintf(fp, L"\n");
		fwprintf(fp, L"#include<LayerErrorCode.h>\n");
		fwprintf(fp, L"#include<INNLayerConfig.h>\n");
		fwprintf(fp, L"#include<INNLayer.h>\n");
		fwprintf(fp, L"\n");
		fwprintf(fp, L"#include<guiddef.h>\n");
		fwprintf(fp, L"\n");
		fwprintf(fp, L"\n");
		fwprintf(fp, L"\n");
		fwprintf(fp, L"\n");
		fwprintf(fp, L"\n");
		fwprintf(fp, L"\n");
		fwprintf(fp, L"#endif // __CUSTOM_DEEP_NN_LAYER_%s_H__\n", fileName.c_str());


		// �t�@�C���N���[�Y
		fclose(fp);
	}

	return 0;
}

