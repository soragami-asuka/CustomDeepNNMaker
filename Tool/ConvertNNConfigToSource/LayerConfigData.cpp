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
int LayerConfigData::ConvertToCPPFile(const boost::filesystem::path& exportDirPath, const std::wstring& fileName)const
{
	return 0;
}

