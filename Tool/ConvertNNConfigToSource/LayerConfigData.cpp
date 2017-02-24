//======================================
// レイヤーの設定情報について記載
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
	/** レイヤー設定情報をXMLから読み込む */
	INNLayerConfigEx* ReadLayerConfigXML(boost::property_tree::ptree& pTree, const GUID& layerCode, const VersionCode& versionCode)
	{
		INNLayerConfigEx* pConfig = CustomDeepNNLibrary::CreateEmptyLayerConfig(layerCode, versionCode);
		if(pConfig == NULL)
			return NULL;

		// 内部のアイテムを読み込む
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
				// 名前
				if(boost::optional<std::string> pValue = it.second.get_optional<std::string>("Name"))
				{
					name = UTF8toUnicode(pValue.get());
				}
				// テキスト
				if(boost::optional<std::string> pValue = it.second.get_optional<std::string>("Text"))
				{
					text = UTF8toUnicode(pValue.get());
				}
				// 最小値
				if(boost::optional<int> pValue = it.second.get_optional<int>("Min"))
				{
					minValue = pValue.get();
				}
				// 最大値
				if(boost::optional<int> pValue = it.second.get_optional<int>("Max"))
				{
					maxValue = pValue.get();
				}
				// デフォルト値
				if(boost::optional<int> pValue = it.second.get_optional<int>("Default"))
				{
					defaultValue = pValue.get();
				}

				// アイテムを作成する
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
				// 名前
				if(boost::optional<std::string> pValue = it.second.get_optional<std::string>("Name"))
				{
					name = UTF8toUnicode(pValue.get());
				}
				// テキスト
				if(boost::optional<std::string> pValue = it.second.get_optional<std::string>("Text"))
				{
					text = UTF8toUnicode(pValue.get());
				}
				// 最小値
				if(boost::optional<float> pValue = it.second.get_optional<float>("Min"))
				{
					minValue = pValue.get();
				}
				// 最大値
				if(boost::optional<float> pValue = it.second.get_optional<float>("Max"))
				{
					maxValue = pValue.get();
				}
				// デフォルト値
				if(boost::optional<float> pValue = it.second.get_optional<float>("Default"))
				{
					defaultValue = pValue.get();
				}

				// アイテムを作成する
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
				// 名前
				if(boost::optional<std::string> pValue = it.second.get_optional<std::string>("Name"))
				{
					name = UTF8toUnicode(pValue.get());
				}
				// テキスト
				if(boost::optional<std::string> pValue = it.second.get_optional<std::string>("Text"))
				{
					text = UTF8toUnicode(pValue.get());
				}
				// デフォルト値
				if(boost::optional<bool> pValue = it.second.get_optional<bool>("Default"))
				{
					defaultValue = pValue.get();
				}

				// アイテムを作成する
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
				// 名前
				if(boost::optional<std::string> pValue = it.second.get_optional<std::string>("Name"))
				{
					name = UTF8toUnicode(pValue.get());
				}
				// テキスト
				if(boost::optional<std::string> pValue = it.second.get_optional<std::string>("Text"))
				{
					text = UTF8toUnicode(pValue.get());
				}
				// デフォルト値
				if(boost::optional<std::string> pValue = it.second.get_optional<std::string>("Default"))
				{
					defaultValue = UTF8toUnicode(pValue.get());
				}

				// アイテムを作成する
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
				// 名前
				if(boost::optional<std::string> pValue = it.second.get_optional<std::string>("Name"))
				{
					name = UTF8toUnicode(pValue.get());
				}
				// テキスト
				if(boost::optional<std::string> pValue = it.second.get_optional<std::string>("Text"))
				{
					text = UTF8toUnicode(pValue.get());
				}
				// デフォルト値
				if(boost::optional<std::string> pValue = it.second.get_optional<std::string>("Default"))
				{
					defaultValue = UTF8toUnicode(pValue.get());
				}

				// アイテムを作成する
				INNLayerConfigItemEx_Enum* pItem = CreateLayerCofigItem_Enum(id.c_str(), name.c_str(), text.c_str());
				if(pItem == NULL)
				{
					delete pConfig;
					return NULL;
				}

				// 列挙値を追加する
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
					// 名前
					if(boost::optional<std::string> pValue = it_item.second.get_optional<std::string>("Name"))
					{
						item_name = UTF8toUnicode(pValue.get());
					}
					// テキスト
					if(boost::optional<std::string> pValue = it_item.second.get_optional<std::string>("Text"))
					{
						item_text = UTF8toUnicode(pValue.get());
					}

					pItem->AddEnumItem(item_id.c_str(), item_name.c_str(), item_text.c_str());
				}

				// デフォルト値を設定
				pItem->SetDefaultItem(defaultValue.c_str());

				// 設定情報に追加
				pConfig->AddItem(pItem);
			}
		}

		return pConfig;
	}

	/** 文字列を改行単位で分離して配列化 */
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
				if(!buf.empty())
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

	/** 文字列の改行を\n→\\n変換して返す */
	std::wstring TextToSingleLine(const std::wstring& input)
	{
		std::wstring result = input;

		std::wstring from = L"\n";
		std::wstring to   = L"\\n";

		std::wstring::size_type pos = result.find(from);
		while(pos != std::wstring::npos)
		{
			result.replace(pos, from.size(), to);
			pos = result.find(from, pos + to.size());
		}

		return result;
	}

	/** データ構造を構造体に変換して出力する */
	int WriteStructureToStructSource(FILE* fp, INNLayerConfig& structure)
	{
		for(unsigned int itemNum=0; itemNum<structure.GetItemCount(); itemNum++)
		{
			auto pItem = structure.GetItemByNum(itemNum);
			if(pItem == NULL)
				continue;

			// 名前
			wchar_t szName[CONFIGITEM_NAME_MAX];
			pItem->GetConfigName(szName);

			// ID
			wchar_t szID[CONFIGITEM_ID_MAX];
			pItem->GetConfigID(szID);

			// 説明文
			wchar_t szText[CONFIGITEM_TEXT_MAX];
			std::vector<std::wstring> lpText;
			pItem->GetConfigText(szText);
			TextToStringArray(szText, lpText);

			fwprintf(fp, L"		/** Name : %ls\n", szName);
			fwprintf(fp, L"		  * ID   : %ls\n", szID);
			if(lpText.size() > 0)
			{
				fwprintf(fp, L"		  * Text : %ls\n", lpText[0].c_str());
				for(unsigned int i=1; i<lpText.size(); i++)
					fwprintf(fp, L"		  *       : %ls\n", lpText[i].c_str());
			}
			fwprintf(fp, L"		  */\n");

			switch(pItem->GetItemType())
			{
			case CONFIGITEM_TYPE_FLOAT:
				{
					const INNLayerConfigItem_Float* pItemFloat = dynamic_cast<const INNLayerConfigItem_Float*>(pItem);
					if(pItemFloat == NULL)
						break;
					fwprintf(fp, L"		float %ls;\n", szID);
				}
				break;
			case CONFIGITEM_TYPE_INT:
				{
					const INNLayerConfigItem_Int* pItemInt = dynamic_cast<const INNLayerConfigItem_Int*>(pItem);
					if(pItemInt == NULL)
						break;
					fwprintf(fp, L"		int %ls;\n", szID);
				}
				break;
			case CONFIGITEM_TYPE_STRING:
				{
					const INNLayerConfigItem_String* pItemString = dynamic_cast<const INNLayerConfigItem_String*>(pItem);
					if(pItemString == NULL)
						break;
					fwprintf(fp, L"		const wchar_t %ls;\n", szID);
				}
				break;
			case CONFIGITEM_TYPE_BOOL:
				{
					const INNLayerConfigItem_Bool* pItemBool = dynamic_cast<const INNLayerConfigItem_Bool*>(pItem);
					if(pItemBool == NULL)
						break;
					fwprintf(fp, L"		const bool %ls;\n", szID);
				}
				break;
			case CONFIGITEM_TYPE_ENUM:
				{
					const INNLayerConfigItem_Enum* pItemEnum = dynamic_cast<const INNLayerConfigItem_Enum*>(pItem);
					if(pItemEnum == NULL)
						break;
					fwprintf(fp, L"		const enum{\n");
					for(unsigned int enumNum=0; enumNum<pItemEnum->GetEnumCount(); enumNum++)
					{
						wchar_t szEnumName[CONFIGITEM_NAME_MAX];
						wchar_t szEnumID[CONFIGITEM_ID_MAX];
						wchar_t szEnumText[CONFIGITEM_TEXT_MAX];
						std::vector<std::wstring> lpEnumText;

						// 名前
						pItemEnum->GetEnumName(enumNum, szEnumName);
						// ID
						pItemEnum->GetEnumID(enumNum, szEnumID);
						// テキスト
						pItemEnum->GetEnumText(enumNum, szEnumText);
						TextToStringArray(szEnumText, lpEnumText);
						

						fwprintf(fp, L"			/** Name : %ls\n", szEnumName);
						fwprintf(fp, L"			  * ID   : %ls\n", szEnumID);
						if(lpText.size() > 0)
						{
							fwprintf(fp, L"			  * Text : %ls\n", lpEnumText[0].c_str());
							for(unsigned int i=1; i<lpText.size(); i++)
							fwprintf(fp, L"			  *      : %ls\n", lpEnumText[i].c_str());
						}
						fwprintf(fp, L"			  */\n");
						fwprintf(fp, L"			%ls_%ls,\n", szID, szEnumID);
						fwprintf(fp, L"\n");
					}
					fwprintf(fp, L"		}%ls;\n", szID);
				}
				break;
			}

			fwprintf(fp, L"\n");
		}
		return 0;
	}

	/** データ構造をソースに変換して出力する */
	int WriteStructureToCreateSource(FILE* fp, INNLayerConfig& structure, const std::wstring& dataCode)
	{
		fwprintf(fp, L"	GUID layerCode;\n");
		fwprintf(fp, L"	GetLayerCode(layerCode);\n");
		fwprintf(fp, L"\n");
		fwprintf(fp, L"	CustomDeepNNLibrary::VersionCode versionCode;\n");
		fwprintf(fp, L"	GetVersionCode(versionCode);\n");
		fwprintf(fp, L"\n");
		fwprintf(fp, L"\n");
		fwprintf(fp, L"	// Create Empty Setting Data\n");
		fwprintf(fp, L"	CustomDeepNNLibrary::INNLayerConfigEx* pLayerConfig = CustomDeepNNLibrary::CreateEmptyLayerConfig(layerCode, versionCode);\n");
		fwprintf(fp, L"	if(pLayerConfig == NULL)\n");
		fwprintf(fp, L"		return NULL;\n");
		fwprintf(fp, L"\n");
		fwprintf(fp, L"\n");
		fwprintf(fp, L"	// Create Item\n");
		for(unsigned int itemNum=0; itemNum<structure.GetItemCount(); itemNum++)
		{
			auto pItem = structure.GetItemByNum(itemNum);
			if(pItem == NULL)
				continue;

			// 名前
			wchar_t szName[CONFIGITEM_NAME_MAX];
			pItem->GetConfigName(szName);

			// ID
			wchar_t szID[CONFIGITEM_ID_MAX];
			pItem->GetConfigID(szID);

			// 説明文
			wchar_t szText[CONFIGITEM_TEXT_MAX];
			std::vector<std::wstring> lpText;
			pItem->GetConfigText(szText);
			TextToStringArray(szText, lpText);

			fwprintf(fp, L"	/** Name : %ls\n", szName);
			fwprintf(fp, L"	  * ID   : %ls\n", szID);
			if(lpText.size() > 0)
			{
				fwprintf(fp, L"	  * Text : %ls\n", lpText[0].c_str());
				for(unsigned int i=1; i<lpText.size(); i++)
					fwprintf(fp, L"	  *      : %ls\n", lpText[i].c_str());
			}
			fwprintf(fp, L"	  */\n");

			switch(pItem->GetItemType())
			{
			case CONFIGITEM_TYPE_FLOAT:
				{
					const INNLayerConfigItem_Float* pItemFloat = dynamic_cast<const INNLayerConfigItem_Float*>(pItem);
					if(pItemFloat == NULL)
						break;
					fwprintf(fp, L"		float %ls;\n", szID);
				}
				break;
			case CONFIGITEM_TYPE_INT:
				{
					const INNLayerConfigItem_Int* pItemInt = dynamic_cast<const INNLayerConfigItem_Int*>(pItem);
					if(pItemInt == NULL)
						break;

					wchar_t szItemID[CustomDeepNNLibrary::CONFIGITEM_ID_MAX];
					pItemInt->GetConfigID(szItemID);

					fwprintf(fp, L"	pLayerConfig->AddItem(\n");
					fwprintf(fp, L"		CustomDeepNNLibrary::CreateLayerCofigItem_Int(\n");
					fwprintf(fp, L"			L\"%ls\",\n", szItemID);
					fwprintf(fp, L"			CurrentLanguage::g_lpItemData_%ls[L\"%ls\"].name.c_str(),\n", dataCode.c_str(), szItemID);
					fwprintf(fp, L"			CurrentLanguage::g_lpItemData_%ls[L\"%ls\"].text.c_str(),\n", dataCode.c_str(), szItemID);
					fwprintf(fp, L"			%ld, %ld, %ld));\n", pItemInt->GetMin(), pItemInt->GetMax(), pItemInt->GetDefault());
				}
				break;
			case CONFIGITEM_TYPE_STRING:
				{
					const INNLayerConfigItem_String* pItemString = dynamic_cast<const INNLayerConfigItem_String*>(pItem);
					if(pItemString == NULL)
						break;
					fwprintf(fp, L"		const wchar_t %ls;\n", szID);
				}
				break;
			case CONFIGITEM_TYPE_BOOL:
				{
					const INNLayerConfigItem_Bool* pItemBool = dynamic_cast<const INNLayerConfigItem_Bool*>(pItem);
					if(pItemBool == NULL)
						break;
					fwprintf(fp, L"		const bool %ls;\n", szID);
				}
				break;
			case CONFIGITEM_TYPE_ENUM:
				{
					const INNLayerConfigItem_Enum* pItemEnum = dynamic_cast<const INNLayerConfigItem_Enum*>(pItem);
					if(pItemEnum == NULL)
						break;
					fwprintf(fp, L"		const enum{\n");
					for(unsigned int enumNum=0; enumNum<pItemEnum->GetEnumCount(); enumNum++)
					{
						wchar_t szEnumName[CONFIGITEM_NAME_MAX];
						wchar_t szEnumID[CONFIGITEM_ID_MAX];
						wchar_t szEnumText[CONFIGITEM_TEXT_MAX];
						std::vector<std::wstring> lpEnumText;

						// 名前
						pItemEnum->GetEnumName(enumNum, szEnumName);
						// ID
						pItemEnum->GetEnumID(enumNum, szEnumID);
						// テキスト
						pItemEnum->GetEnumText(enumNum, szEnumText);
						TextToStringArray(szEnumText, lpEnumText);
						

						fwprintf(fp, L"			/** Name : %ls\n", szEnumName);
						fwprintf(fp, L"			  * ID   : %ls\n", szEnumID);
						if(lpText.size() > 0)
						{
							fwprintf(fp, L"			  * Text : %ls\n", lpEnumText[0].c_str());
							for(unsigned int i=1; i<lpText.size(); i++)
							fwprintf(fp, L"			  *      : %ls\n", lpEnumText[i].c_str());
						}
						fwprintf(fp, L"			  */\n");
						fwprintf(fp, L"			%ls_%ls,\n", szID, szEnumID);
						fwprintf(fp, L"\n");
					}
					fwprintf(fp, L"		}%ls;\n", szID);
				}
				break;
			}

			fwprintf(fp, L"\n");
		}
		return 0;
	}
}


/** コンストラクタ */
LayerConfigData::LayerConfigData()
	:	guid	()					/**< 識別コード */
	,	default_language	(L"")	/**< 基本言語 */

	,	name	(L"")	/**< 名前 */
	,	text	(L"")	/**< 説明テキスト */

	,	pStructure	(NULL)	/**< レイヤー構造定義情報 */
	,	pLearn		(NULL)	/**< 学習設定情報 */
{
}
/** デストラクタ */
LayerConfigData::~LayerConfigData()
{
	if(this->pStructure)
		delete this->pStructure;
	if(this->pLearn)
		delete this->pLearn;
}

/** XMLファイルから情報を読み込む.
	@param	configFilePath	読み込むXMLファイルのパス
	@return	成功した場合0が返る. */
int LayerConfigData::ReadFromXMLFile(const boost::filesystem::wpath& configFilePath)
{
	// 現状のファイルを初期化
	if(this->pStructure)
		delete this->pStructure;
	if(this->pLearn)
		delete this->pLearn;


	// XMLファイルの読み込み
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

	// 情報を読み込む
	try
	{
		// GUID
		if(boost::optional<std::string> guid_str = pXmlTree.get_optional<std::string>("Config.<xmlattr>.guid"))
		{
			std::wstring buf = UTF8toUnicode(guid_str.get());

			// 文字列を分解
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

		// バージョン
		if(boost::optional<std::string> version_str = pXmlTree.get_optional<std::string>("Config.<xmlattr>.version"))
		{
			std::wstring buf = UTF8toUnicode(version_str.get());

			// 文字列を分解
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

		// 基本言語
		if(boost::optional<std::string> language_str = pXmlTree.get_optional<std::string>("Config.<xmlattr>.default-language"))
		{
			this->default_language = UTF8toUnicode(language_str.get());
		}
		else
		{
			this->default_language = L"ja";
		}

		// 名前
		if(boost::optional<std::string> name_str = pXmlTree.get_optional<std::string>("Config.Name"))
		{
			this->name = UTF8toUnicode(name_str.get());
		}
		else
		{
			this->name = L"";
		}
		
		// 説明文
		if(boost::optional<std::string> text_str = pXmlTree.get_optional<std::string>("Config.Text"))
		{
			this->text = UTF8toUnicode(text_str.get());
		}
		else
		{
			this->text = L"";
		}

		// レイヤー構造
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

		// 学習設定
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
/** C++言語ソースファイルに変換/出力する.
	.h/.cppファイルが生成される.
	@param	exportDirPath	出力先ディレクトリパス
	@param	fileName		出力ファイル名.拡張子は除く.
	@return 成功した場合0が返る. */
int LayerConfigData::ConvertToCPPFile(const boost::filesystem::wpath& exportDirPath, const std::wstring& fileName)const
{
	// 出力言語設定を変更
	_wsetlocale(LC_ALL, this->default_language.c_str());

	boost::filesystem::wpath funcSourceFilePath = exportDirPath / (fileName + L"_FUNC" + L".cpp");
	boost::filesystem::wpath funcHeaderFilePath = exportDirPath / (fileName + L"_FUNC" + L".hpp");
	boost::filesystem::wpath dataHeaderFilePath = exportDirPath / (fileName + L"_DATA" + L".hpp");

	// GUIDを文字列に変換
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

	//======================
	// データヘッダファイルの出力
	//======================
	{
		// ファイルオープン
		FILE* fp = _wfopen(dataHeaderFilePath.wstring().c_str(), L"w");
		if(fp == NULL)
			return -1;

		fwprintf(fp, L"/*--------------------------------------------\n");
		fwprintf(fp, L" * FileName  : %ls\n", dataHeaderFilePath.filename().wstring().c_str());
		fwprintf(fp, L" * LayerName : %ls\n", this->name.c_str());
		fwprintf(fp, L" * guid      : %ls\n", strGUID.c_str());
		{
			std::vector<std::wstring> lpText;
			TextToStringArray(this->text, lpText);
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
		}
		fwprintf(fp, L"--------------------------------------------*/\n");
		fwprintf(fp, L"#ifndef __CUSTOM_DEEP_NN_LAYER_DATA_%s_H__\n", fileName.c_str());
		fwprintf(fp, L"#define __CUSTOM_DEEP_NN_LAYER_DATA_%s_H__\n", fileName.c_str());
		fwprintf(fp, L"\n");
		fwprintf(fp, L"#include<guiddef.h>\n");
		fwprintf(fp, L"\n");
		fwprintf(fp, L"#include<LayerErrorCode.h>\n");
		fwprintf(fp, L"#include<INNLayerConfig.h>\n");
		fwprintf(fp, L"#include<INNLayer.h>\n");
		fwprintf(fp, L"\n");
		fwprintf(fp, L"namespace CustomDeepNNLibrary {\n");
		fwprintf(fp, L"namespace %s {\n", fileName.c_str());
		fwprintf(fp, L"\n");
		if(this->pStructure->GetItemCount() > 0)
		{
			fwprintf(fp, L"	/** Layer structure */\n");
			fwprintf(fp, L"	struct LayerStructure\n");
			fwprintf(fp, L"	{\n");
			WriteStructureToStructSource(fp, *this->pStructure);
			fwprintf(fp, L"	};\n");
			fwprintf(fp, L"\n");
		}
		if(this->pLearn->GetItemCount() > 0)
		{
			fwprintf(fp, L"	/** Learning data structure */\n");
			fwprintf(fp, L"	struct LearnDataStructure\n");
			fwprintf(fp, L"	{\n");
			WriteStructureToStructSource(fp, *this->pLearn);
			fwprintf(fp, L"	};\n");
			fwprintf(fp, L"\n");
		}
		fwprintf(fp, L"} // %s\n", fileName.c_str());
		fwprintf(fp, L"} // CustomDeepNNLibrary\n");
		fwprintf(fp, L"\n");
		fwprintf(fp, L"\n");
		fwprintf(fp, L"#endif // __CUSTOM_DEEP_NN_LAYER_DATA_%s_H__\n", fileName.c_str());

		// ファイルクローズ
		fclose(fp);
	}

	//======================
	// 関数ヘッダファイルの出力
	//======================
	{
		// ファイルオープン
		FILE* fp = _wfopen(funcHeaderFilePath.wstring().c_str(), L"w");
		if(fp == NULL)
			return -1;

		fwprintf(fp, L"/*--------------------------------------------\n");
		fwprintf(fp, L" * FileName  : %ls\n", dataHeaderFilePath.filename().wstring().c_str());
		fwprintf(fp, L" * LayerName : %ls\n", this->name.c_str());
		fwprintf(fp, L" * guid      : %ls\n", strGUID.c_str());
		{
			std::vector<std::wstring> lpText;
			TextToStringArray(this->text, lpText);
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
		}
		fwprintf(fp, L"--------------------------------------------*/\n");
		fwprintf(fp, L"#ifndef __CUSTOM_DEEP_NN_LAYER_FUNC_%s_H__\n", fileName.c_str());
		fwprintf(fp, L"#define __CUSTOM_DEEP_NN_LAYER_FUNC_%s_H__\n", fileName.c_str());
		fwprintf(fp, L"\n");
		fwprintf(fp, L"#define EXPORT_API extern \"C\" __declspec(dllexport)\n");
		fwprintf(fp, L"\n");
		fwprintf(fp, L"#include<guiddef.h>\n");
		fwprintf(fp, L"\n");
		fwprintf(fp, L"#include<LayerErrorCode.h>\n");
		fwprintf(fp, L"#include<INNLayerConfig.h>\n");
		fwprintf(fp, L"#include<INNLayer.h>\n");
		fwprintf(fp, L"\n");
		fwprintf(fp, L"#include\"%ls\"\n", dataHeaderFilePath.filename().wstring().c_str());
		fwprintf(fp, L"\n");
		fwprintf(fp, L"\n");
		fwprintf(fp, L"/** Acquire the layer identification code.\n");
		fwprintf(fp, L"  * @param  o_layerCode    Storage destination buffer.\n");
		fwprintf(fp, L"  * @return On success 0. \n");
		fwprintf(fp, L"  */\n");
		fwprintf(fp, L"EXPORT_API CustomDeepNNLibrary::ELayerErrorCode GetLayerCode(GUID& o_layerCode);\n");
		fwprintf(fp, L"\n");
		fwprintf(fp, L"/** Get version code.\n");
		fwprintf(fp, L"  * @param  o_versionCode    Storage destination buffer.\n");
		fwprintf(fp, L"  * @return On success 0. \n");
		fwprintf(fp, L"  */\n");
		fwprintf(fp, L"EXPORT_API CustomDeepNNLibrary::ELayerErrorCode GetVersionCode(CustomDeepNNLibrary::VersionCode& o_versionCode);\n");
		fwprintf(fp, L"\n");
		fwprintf(fp, L"\n");
		fwprintf(fp, L"/** Create a layer structure setting.\n");
		fwprintf(fp, L"  * @return If successful, new configuration information.\n");
		fwprintf(fp, L"  */\n");
		fwprintf(fp, L"EXPORT_API CustomDeepNNLibrary::INNLayerConfig* CreateLayerStructureSetting(void);\n");
		fwprintf(fp, L"\n");
		fwprintf(fp, L"/** Create layer structure settings from buffer.\n");
		fwprintf(fp, L"  * @param  i_lpBuffer       Start address of the read buffer.\n");
		fwprintf(fp, L"  * @param  i_bufferSize     The size of the readable buffer.\n");
		fwprintf(fp, L"  * @param  o_useBufferSize  Buffer size actually read.\n");
		fwprintf(fp, L"  * @return If successful, the configuration information created from the buffer\n");
		fwprintf(fp, L"  */\n");
		fwprintf(fp, L"EXPORT_API CustomDeepNNLibrary::INNLayerConfig* CreateLayerStructureSettingFromBuffer(const BYTE* i_lpBuffer, int i_bufferSize, int& o_useBufferSize);\n");
		fwprintf(fp, L"\n");
		fwprintf(fp, L"\n");
		fwprintf(fp, L"/** Create a learning setting.\n");
		fwprintf(fp, L"  * @return If successful, new configuration information. */\n");
		fwprintf(fp, L"EXPORT_API CustomDeepNNLibrary::INNLayerConfig* CreateLearningSetting(void);\n");
		fwprintf(fp, L"\n");
		fwprintf(fp, L"/** Create learning settings from buffer.\n");
		fwprintf(fp, L"  * @param  i_lpBuffer       Start address of the read buffer.\n");
		fwprintf(fp, L"  * @param  i_bufferSize     The size of the readable buffer.\n");
		fwprintf(fp, L"  * @param  o_useBufferSize  Buffer size actually read.\n");
		fwprintf(fp, L"  * @return If successful, the configuration information created from the buffer\n");
		fwprintf(fp, L"  */\n");
		fwprintf(fp, L"EXPORT_API CustomDeepNNLibrary::INNLayerConfig* CreateLearningSettingFromBuffer(const BYTE* i_lpBuffer, int i_bufferSize, int& o_useBufferSize);\n");
		fwprintf(fp, L"\n");
		fwprintf(fp, L"\n");
		fwprintf(fp, L"/** Create a layer for CPU processing.\n");
		fwprintf(fp, L"  * @param GUID of layer to create.\n");
		fwprintf(fp, L"  */\n");
		fwprintf(fp, L"EXPORT_API CustomDeepNNLibrary::INNLayer* CreateLayerCPU(GUID guid);\n");
		fwprintf(fp, L"\n");
		fwprintf(fp, L"/** Create a layer for GPU processing.\n");
		fwprintf(fp, L"  * @param GUID of layer to create.\n");
		fwprintf(fp, L"  */\n");
		fwprintf(fp, L"EXPORT_API CustomDeepNNLibrary::INNLayer* CreateLayerGPU(GUID guid);\n");
		fwprintf(fp, L"\n");
		fwprintf(fp, L"\n");
		fwprintf(fp, L"\n");
		fwprintf(fp, L"#endif // __CUSTOM_DEEP_NN_LAYER_FUNC_%s_H__\n", fileName.c_str());
	}

	//======================
	// 関数ソースファイルの出力
	//======================
	{
		// ファイルオープン
		FILE* fp = _wfopen(funcSourceFilePath.wstring().c_str(), L"w");
		if(fp == NULL)
			return -1;

		fwprintf(fp, L"/*--------------------------------------------\n");
		fwprintf(fp, L" * FileName  : %ls\n", funcSourceFilePath.filename().wstring().c_str());
		fwprintf(fp, L" * LayerName : %ls\n", this->name.c_str());
		fwprintf(fp, L" * guid      : %ls\n", strGUID.c_str());
		{
			std::vector<std::wstring> lpText;
			TextToStringArray(this->text, lpText);
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
		}
		fwprintf(fp, L"--------------------------------------------*/\n");
		fwprintf(fp, L"#include\"stdafx.h\"\n");
		fwprintf(fp, L"\n");
		fwprintf(fp, L"#include<guiddef.h>\n");
		fwprintf(fp, L"#include<string>\n");
		fwprintf(fp, L"#include<map>\n");
		fwprintf(fp, L"\n");
		fwprintf(fp, L"#include<NNLayerConfig.h>\n");
		fwprintf(fp, L"\n");
		fwprintf(fp, L"#include\"%ls\"\n", funcHeaderFilePath.filename().wstring().c_str());
		fwprintf(fp, L"\n");
		fwprintf(fp, L"\n");
		fwprintf(fp, L"// {%08X-%04X-%04X-%02X%02X-%02X%02X%02X%02X%02X%02X}\n",
			this->guid.Data1,
			this->guid.Data2,
			this->guid.Data3,
			this->guid.Data4[0], this->guid.Data4[1],
			this->guid.Data4[2], this->guid.Data4[3], this->guid.Data4[4], this->guid.Data4[5], this->guid.Data4[6], this->guid.Data4[7]);
		fwprintf(fp, L"static const GUID g_guid = {0x%08x, 0x%04x, 0x%04x, {0x%02x, 0x%02x, 0x%02x, 0x%02x, 0x%02x, 0x%02x, 0x%02x, 0x%02x}};\n",
			this->guid.Data1,
			this->guid.Data2,
			this->guid.Data3,
			this->guid.Data4[0], this->guid.Data4[1],
			this->guid.Data4[2], this->guid.Data4[3], this->guid.Data4[4], this->guid.Data4[5], this->guid.Data4[6], this->guid.Data4[7]);
		fwprintf(fp, L"\n");
		fwprintf(fp, L"// VersionCode\n");
		fwprintf(fp, L"static const CustomDeepNNLibrary::VersionCode g_version = { %3d, %3d, %3d, %3d}; \n",
			this->version.lpData[0], this->version.lpData[1], this->version.lpData[2], this->version.lpData[3]);
		fwprintf(fp, L"\n");
		fwprintf(fp, L"\n");
		fwprintf(fp, L"\n");
		fwprintf(fp, L"struct StringData\n");
		fwprintf(fp, L"{\n");
		fwprintf(fp, L"    std::wstring name;\n");
		fwprintf(fp, L"    std::wstring text;\n");
		fwprintf(fp, L"};\n");
		fwprintf(fp, L"\n");
		fwprintf(fp, L"namespace DefaultLanguage\n");
		fwprintf(fp, L"{\n");
		fwprintf(fp, L"    /** Language Code */\n");
		fwprintf(fp, L"    static const std::wstring g_languageCode = L\"%ls\";\n", this->default_language.c_str());
		fwprintf(fp, L"\n");
		fwprintf(fp, L"    /** Base */\n");
		fwprintf(fp, L"    static const StringData g_baseData = \n");
		fwprintf(fp, L"    {\n");
		fwprintf(fp, L"        L\"%ls\",\n", this->name.c_str());
		fwprintf(fp, L"        L\"%ls\"\n", TextToSingleLine(this->text).c_str());
		fwprintf(fp, L"    };\n");
		fwprintf(fp, L"\n");
		fwprintf(fp, L"\n");
		// レイヤー構造
		{
			fwprintf(fp, L"    /** ItemData Layer Structure <id, StringData> */\n");
			fwprintf(fp, L"    static const std::map<std::wstring, StringData> g_lpItemData_LayerStructure = \n");
			fwprintf(fp, L"    {\n");
			for(unsigned int itemNum=0; itemNum<this->pStructure->GetItemCount(); itemNum++)
			{
				auto pItem = this->pStructure->GetItemByNum(itemNum);
				if(pItem == NULL)
					continue;

				wchar_t szID[CustomDeepNNLibrary::CONFIGITEM_ID_MAX];
				wchar_t szName[CustomDeepNNLibrary::CONFIGITEM_NAME_MAX];
				wchar_t szText[CustomDeepNNLibrary::CONFIGITEM_TEXT_MAX];

				pItem->GetConfigID(szID);
				pItem->GetConfigName(szName);
				pItem->GetConfigText(szText);


				fwprintf(fp, L"        {\n");
				fwprintf(fp, L"            L\"%ls\",\n", szID);
				fwprintf(fp, L"            {\n");
				fwprintf(fp, L"                L\"%ls\",\n", szName);
				fwprintf(fp, L"                L\"%ls\",\n", TextToSingleLine(szText).c_str());
				fwprintf(fp, L"            }\n");
				fwprintf(fp, L"        },\n");
			}
			fwprintf(fp, L"    };\n");
			fwprintf(fp, L"\n");
			fwprintf(fp, L"\n");
			fwprintf(fp, L"    /** ItemData Layer Structure Enum <id, enumID, StringData> */\n");
			fwprintf(fp, L"    static const std::map<std::wstring, std::map<std::wstring, StringData>> g_lpItemDataEnum_LayerStructure =\n");
			fwprintf(fp, L"    {\n");
			for(unsigned int itemNum=0; itemNum<this->pStructure->GetItemCount(); itemNum++)
			{
				auto pItem = this->pStructure->GetItemByNum(itemNum);
				if(pItem == NULL)
					continue;

				if(pItem->GetItemType() == CustomDeepNNLibrary::NNLayerConfigItemType::CONFIGITEM_TYPE_ENUM)
				{
					const CustomDeepNNLibrary::INNLayerConfigItem_Enum* pItemEnum = dynamic_cast<const CustomDeepNNLibrary::INNLayerConfigItem_Enum*>(pItem);
					
					wchar_t szID[CustomDeepNNLibrary::CONFIGITEM_ID_MAX];
					pItem->GetConfigID(szID);

					fwprintf(fp, L"        {\n");
					fwprintf(fp, L"            L\"%ls\",\n", szID);
					fwprintf(fp, L"            {\n");
					for(unsigned int enumItemNum=0; enumItemNum<pItemEnum->GetEnumCount(); enumItemNum++)
					{					
						wchar_t szEnumID[CustomDeepNNLibrary::CONFIGITEM_ID_MAX];
						wchar_t szEnumName[CustomDeepNNLibrary::CONFIGITEM_NAME_MAX];
						wchar_t szEnumText[CustomDeepNNLibrary::CONFIGITEM_TEXT_MAX];

						pItemEnum->GetEnumID(enumItemNum, szEnumID);
						pItemEnum->GetEnumName(enumItemNum, szEnumName);
						pItemEnum->GetEnumText(enumItemNum, szEnumText);

						fwprintf(fp, L"                {\n");
						fwprintf(fp, L"                    L\"%ls\",\n", szEnumID);
						fwprintf(fp, L"                    {\n");
						fwprintf(fp, L"                        L\"%ls\",\n", szEnumName);
						fwprintf(fp, L"                        L\"%ls\",\n", TextToSingleLine(szEnumText).c_str());
						fwprintf(fp, L"                    },\n");
						fwprintf(fp, L"                },\n");
					}
					fwprintf(fp, L"            }\n");
					fwprintf(fp, L"        },\n");
				}
			}
			fwprintf(fp, L"    };\n");
			fwprintf(fp, L"\n");
			fwprintf(fp, L"\n");
		}
		fwprintf(fp, L"\n");
		// 学習データ
		{
			fwprintf(fp, L"    /** ItemData Learn <id, StringData> */\n");
			fwprintf(fp, L"    static const std::map<std::wstring, StringData> g_lpItemData_Learn = \n");
			fwprintf(fp, L"    {\n");
			for(unsigned int itemNum=0; itemNum<this->pLearn->GetItemCount(); itemNum++)
			{
				auto pItem = this->pLearn->GetItemByNum(itemNum);
				if(pItem == NULL)
					continue;

				wchar_t szID[CustomDeepNNLibrary::CONFIGITEM_ID_MAX];
				wchar_t szName[CustomDeepNNLibrary::CONFIGITEM_NAME_MAX];
				wchar_t szText[CustomDeepNNLibrary::CONFIGITEM_TEXT_MAX];

				pItem->GetConfigID(szID);
				pItem->GetConfigName(szName);
				pItem->GetConfigText(szText);


				fwprintf(fp, L"        {\n");
				fwprintf(fp, L"            L\"%ls\",\n", szID);
				fwprintf(fp, L"            {\n");
				fwprintf(fp, L"                L\"%ls\",\n", szName);
				fwprintf(fp, L"                L\"%ls\",\n", TextToSingleLine(szText).c_str());
				fwprintf(fp, L"            }\n");
				fwprintf(fp, L"        },\n");
			}
			fwprintf(fp, L"    };\n");
			fwprintf(fp, L"\n");
			fwprintf(fp, L"\n");
			fwprintf(fp, L"    /** ItemData Learn Enum <id, enumID, StringData> */\n");
			fwprintf(fp, L"    static const std::map<std::wstring, std::map<std::wstring, StringData>> g_lpItemDataEnum_Learn =\n");
			fwprintf(fp, L"    {\n");
			for(unsigned int itemNum=0; itemNum<this->pLearn->GetItemCount(); itemNum++)
			{
				auto pItem = this->pLearn->GetItemByNum(itemNum);
				if(pItem == NULL)
					continue;

				if(pItem->GetItemType() == CustomDeepNNLibrary::NNLayerConfigItemType::CONFIGITEM_TYPE_ENUM)
				{
					const CustomDeepNNLibrary::INNLayerConfigItem_Enum* pItemEnum = dynamic_cast<const CustomDeepNNLibrary::INNLayerConfigItem_Enum*>(pItem);
					
					wchar_t szID[CustomDeepNNLibrary::CONFIGITEM_ID_MAX];
					pItem->GetConfigID(szID);

					fwprintf(fp, L"        {\n");
					fwprintf(fp, L"            L\"%ls\",\n", szID);
					fwprintf(fp, L"            {\n");
					for(unsigned int enumItemNum=0; enumItemNum<pItemEnum->GetEnumCount(); enumItemNum++)
					{					
						wchar_t szEnumID[CustomDeepNNLibrary::CONFIGITEM_ID_MAX];
						wchar_t szEnumName[CustomDeepNNLibrary::CONFIGITEM_NAME_MAX];
						wchar_t szEnumText[CustomDeepNNLibrary::CONFIGITEM_TEXT_MAX];

						pItemEnum->GetEnumID(enumItemNum, szEnumID);
						pItemEnum->GetEnumName(enumItemNum, szEnumName);
						pItemEnum->GetEnumText(enumItemNum, szEnumText);

						fwprintf(fp, L"                {\n");
						fwprintf(fp, L"                    L\"%ls\",\n", szEnumID);
						fwprintf(fp, L"                    {\n");
						fwprintf(fp, L"                        L\"%ls\",\n", szEnumName);
						fwprintf(fp, L"                        L\"%ls\",\n", TextToSingleLine(szEnumText).c_str());
						fwprintf(fp, L"                    },\n");
						fwprintf(fp, L"                },\n");
					}
					fwprintf(fp, L"            }\n");
					fwprintf(fp, L"        },\n");
				}
			}
			fwprintf(fp, L"    };\n");
			fwprintf(fp, L"\n");
			fwprintf(fp, L"\n");
		}
		fwprintf(fp, L"}\n");
		fwprintf(fp, L"\n");
		fwprintf(fp, L"\n");
		fwprintf(fp, L"namespace CurrentLanguage\n");
		fwprintf(fp, L"{\n");
		fwprintf(fp, L"    /** Language Code */\n");
		fwprintf(fp, L"    static const std::wstring g_languageCode = DefaultLanguage::g_languageCode;\n");
		fwprintf(fp, L"\n");
		fwprintf(fp, L"\n");
		fwprintf(fp, L"    /** Base */\n");
		fwprintf(fp, L"    static StringData g_baseData = DefaultLanguage::g_baseData;\n");
		fwprintf(fp, L"\n");
		fwprintf(fp, L"\n");
		// レイヤー構造
		fwprintf(fp, L"    /** ItemData Layer Structure <id, StringData> */\n");
		fwprintf(fp, L"    static std::map<std::wstring, StringData> g_lpItemData_LayerStructure = DefaultLanguage::g_lpItemData_LayerStructure;\n");
		fwprintf(fp, L"\n");
		fwprintf(fp, L"    /** ItemData Learn Enum <id, enumID, StringData> */\n");
		fwprintf(fp, L"    static const std::map<std::wstring, std::map<std::wstring, StringData>> g_lpItemDataEnum_LayerStructure = DefaultLanguage::g_lpItemDataEnum_LayerStructure;\n");
		fwprintf(fp, L"\n");
		fwprintf(fp, L"\n");
		// 学習設定
		fwprintf(fp, L"    /** ItemData Learn <id, StringData> */\n");
		fwprintf(fp, L"    static std::map<std::wstring, StringData> g_lpItemData_Learn = DefaultLanguage::g_lpItemData_Learn;\n");
		fwprintf(fp, L"\n");
		fwprintf(fp, L"    /** ItemData Learn Enum <id, enumID, StringData> */\n");
		fwprintf(fp, L"    static const std::map<std::wstring, std::map<std::wstring, StringData>> g_lpItemDataEnum_Learn = DefaultLanguage::g_lpItemDataEnum_Learn;\n");
		fwprintf(fp, L"\n");
		fwprintf(fp, L"}\n");

		fwprintf(fp, L"\n");
		fwprintf(fp, L"\n");
		fwprintf(fp, L"\n");
		
		fwprintf(fp, L"/** Acquire the layer identification code.\n");
		fwprintf(fp, L"  * @param  o_layerCode    Storage destination buffer.\n");
		fwprintf(fp, L"  * @return On success 0. \n");
		fwprintf(fp, L"  */\n");
		fwprintf(fp, L"EXPORT_API CustomDeepNNLibrary::ELayerErrorCode GetLayerCode(GUID& o_layerCode)\n");
		fwprintf(fp, L"{\n");
		fwprintf(fp, L"	o_layerCode = g_guid;\n");
		fwprintf(fp, L"\n");
		fwprintf(fp, L"	return CustomDeepNNLibrary::LAYER_ERROR_NONE;\n");
		fwprintf(fp, L"}\n");
		fwprintf(fp, L"\n");
		fwprintf(fp, L"/** Get version code.\n");
		fwprintf(fp, L"  * @param  o_versionCode    Storage destination buffer.\n");
		fwprintf(fp, L"  * @return On success 0. \n");
		fwprintf(fp, L"  */\n");
		fwprintf(fp, L"EXPORT_API CustomDeepNNLibrary::ELayerErrorCode GetVersionCode(CustomDeepNNLibrary::VersionCode& o_versionCode)\n");
		fwprintf(fp, L"{\n");
		fwprintf(fp, L"	o_versionCode = g_version;\n");
		fwprintf(fp, L"\n");
		fwprintf(fp, L"	return CustomDeepNNLibrary::LAYER_ERROR_NONE;\n");
		fwprintf(fp, L"}\n");
		fwprintf(fp, L"\n");
		fwprintf(fp, L"\n");
		fwprintf(fp, L"/** Create a layer structure setting.\n");
		fwprintf(fp, L"  * @return If successful, new configuration information.\n");
		fwprintf(fp, L"  */\n");
		fwprintf(fp, L"EXPORT_API CustomDeepNNLibrary::INNLayerConfig* CreateLayerStructureSetting(void)\n");
		fwprintf(fp, L"{\n");
		::WriteStructureToCreateSource(fp, *this->pStructure, L"LayerStructure");
		fwprintf(fp, L"}\n");
		fwprintf(fp, L"\n");
		fwprintf(fp, L"/** Create layer structure settings from buffer.\n");
		fwprintf(fp, L"  * @param  i_lpBuffer       Start address of the read buffer.\n");
		fwprintf(fp, L"  * @param  i_bufferSize     The size of the readable buffer.\n");
		fwprintf(fp, L"  * @param  o_useBufferSize  Buffer size actually read.\n");
		fwprintf(fp, L"  * @return If successful, the configuration information created from the buffer\n");
		fwprintf(fp, L"  */\n");
		fwprintf(fp, L"EXPORT_API CustomDeepNNLibrary::INNLayerConfig* CreateLayerStructureSettingFromBuffer(const BYTE* i_lpBuffer, int i_bufferSize, int& o_useBufferSize)\n");
		fwprintf(fp, L"{\n");
		fwprintf(fp, L"	CustomDeepNNLibrary::INNLayerConfigEx* pLayerConfig = (CustomDeepNNLibrary::INNLayerConfigEx*)CreateLayerStructureSetting();\n");
		fwprintf(fp, L"	if(pLayerConfig == NULL)\n");
		fwprintf(fp, L"		return NULL;\n");
		fwprintf(fp, L"\n");
		fwprintf(fp, L"	int useBufferSize = pLayerConfig->ReadFromBuffer(i_lpBuffer, i_bufferSize);\n");
		fwprintf(fp, L"	if(useBufferSize < 0)\n");
		fwprintf(fp, L"	{\n");
		fwprintf(fp, L"		delete pLayerConfig;\n");
		fwprintf(fp, L"\n");
		fwprintf(fp, L"		return NULL;\n");
		fwprintf(fp, L"	}\n");
		fwprintf(fp, L"	o_useBufferSize = useBufferSize;\n");
		fwprintf(fp, L"\n");
		fwprintf(fp, L"	return pLayerConfig;\n");
		fwprintf(fp, L"}\n");
		fwprintf(fp, L"\n");
		fwprintf(fp, L"\n");
		fwprintf(fp, L"/** Create a learning setting.\n");
		fwprintf(fp, L"  * @return If successful, new configuration information. */\n");
		fwprintf(fp, L"EXPORT_API CustomDeepNNLibrary::INNLayerConfig* CreateLearningSetting(void);\n");
		fwprintf(fp, L"\n");
		fwprintf(fp, L"/** Create learning settings from buffer.\n");
		fwprintf(fp, L"  * @param  i_lpBuffer       Start address of the read buffer.\n");
		fwprintf(fp, L"  * @param  i_bufferSize     The size of the readable buffer.\n");
		fwprintf(fp, L"  * @param  o_useBufferSize  Buffer size actually read.\n");
		fwprintf(fp, L"  * @return If successful, the configuration information created from the buffer\n");
		fwprintf(fp, L"  */\n");
		fwprintf(fp, L"EXPORT_API CustomDeepNNLibrary::INNLayerConfig* CreateLearningSettingFromBuffer(const BYTE* i_lpBuffer, int i_bufferSize, int& o_useBufferSize)\n");
		fwprintf(fp, L"{\n");
		fwprintf(fp, L"	CustomDeepNNLibrary::INNLayerConfigEx* pLayerConfig = (CustomDeepNNLibrary::INNLayerConfigEx*)CreateLearningSetting();\n");
		fwprintf(fp, L"	if(pLayerConfig == NULL)\n");
		fwprintf(fp, L"		return NULL;\n");
		fwprintf(fp, L"\n");
		fwprintf(fp, L"	int useBufferSize = pLayerConfig->ReadFromBuffer(i_lpBuffer, i_bufferSize);\n");
		fwprintf(fp, L"	if(useBufferSize < 0)\n");
		fwprintf(fp, L"	{\n");
		fwprintf(fp, L"		delete pLayerConfig;\n");
		fwprintf(fp, L"\n");
		fwprintf(fp, L"		return NULL;\n");
		fwprintf(fp, L"	}\n");
		fwprintf(fp, L"	o_useBufferSize = useBufferSize;\n");
		fwprintf(fp, L"\n");
		fwprintf(fp, L"	return pLayerConfig;\n");
		fwprintf(fp, L"}\n");
		fwprintf(fp, L"\n");
		fwprintf(fp, L"\n");


		// ファイルクローズ
		fclose(fp);
	}

	return 0;
}

