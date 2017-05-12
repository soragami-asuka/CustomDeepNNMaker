//=========================================
// XML形式で記述されたネットワーク構文を解析する
//=========================================
#include"stdafx.h"

#include<set>;

#include<boost/filesystem.hpp>
#include<boost/property_tree/ptree.hpp>
#include<boost/property_tree/xml_parser.hpp>
#include<boost/regex.hpp>

#include"../../Common/StringUtility/StringUtility.h"
#include"Layer/NeuralNetwork/INNLayerConnectData.h"

#include"XMLParser.h"

using namespace StringUtility;

namespace
{
	std::wstring GUID2WString(const Gravisbell::GUID& i_guid)
	{
		wchar_t szBuf[64];
		swprintf_s(szBuf, L"%08X-%04X-%04X-%02X%02X-%02X%02X%02X%02X%02X%02X",
			i_guid.Data1,
			i_guid.Data2,
			i_guid.Data3,
			i_guid.Data4[0], i_guid.Data4[1],
			i_guid.Data4[2], i_guid.Data4[3], i_guid.Data4[4], i_guid.Data4[5], i_guid.Data4[6], i_guid.Data4[7]);

		return szBuf;
	}
	std::string GUID2String(const Gravisbell::GUID& i_guid)
	{
		return UnicodeToShiftjis(GUID2WString(i_guid));
	}
	Gravisbell::GUID String2GUID(const std::wstring& i_buf)
	{
		// 文字列を分解
		boost::wregex reg(L"^([0-9a-fA-F]{8})-([0-9a-fA-F]{4})-([0-9a-fA-F]{4})-([0-9a-fA-F]{2})([0-9a-fA-F]{2})-([0-9a-fA-F]{2})([0-9a-fA-F]{2})([0-9a-fA-F]{2})([0-9a-fA-F]{2})([0-9a-fA-F]{2})([0-9a-fA-F]{2})$");
		boost::wsmatch match;
		if(boost::regex_search(i_buf, match, reg))
		{
			Gravisbell::GUID guid;

			guid.Data1 = (unsigned long)wcstoul(match[1].str().c_str(), NULL, 16);
			guid.Data2 = (unsigned short)wcstoul(match[2].str().c_str(), NULL, 16);
			guid.Data3 = (unsigned short)wcstoul(match[3].str().c_str(), NULL, 16);
			for(int i=0; i<8; i++)
			{
				guid.Data4[i] = (unsigned char)wcstoul(match[4+i].str().c_str(), NULL, 16);
			}

			return guid;
		}
		else
		{
			return Gravisbell::GUID();
		}
	}
}


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {
namespace Parser {

	namespace
	{
		/** レイヤーデータをXMLファイルに書き出す. */
		Gravisbell::ErrorCode SaveLayerToXML(INNLayerData& i_NNLayer, std::set<Gravisbell::GUID>& lpAlreadyExportLayerDataGUID, const boost::filesystem::wpath& i_layerDirPath, const boost::filesystem::wpath& i_layerFilePath)
		{
			// 出力レイヤーがレイヤー接続型であることを確認
			INNLayerConnectData* pConnectLayerData = dynamic_cast<INNLayerConnectData*>(&i_NNLayer);

			// 出力先ファイルが書き込める状態か確認する
			boost::filesystem::wpath layerFilePath = "";
			if(i_layerFilePath.empty())
			{
				if(pConnectLayerData == NULL)
				{
					layerFilePath = i_layerDirPath / (::GUID2WString(i_NNLayer.GetGUID()) + L".xml");
				}
				else
				{
					layerFilePath = i_layerDirPath / (::GUID2WString(i_NNLayer.GetGUID()) + L".bin");
				}
			}
			else
			{
				layerFilePath = i_layerFilePath;
				if(!boost::filesystem::is_directory(layerFilePath.parent_path()))
				{
					layerFilePath = i_layerDirPath / i_layerFilePath;
				}
			}

			// 自身を保存済みに
			lpAlreadyExportLayerDataGUID.insert(i_NNLayer.GetGUID());

			if(pConnectLayerData)
			{
				// 接続型の場合

				// ツリーを作成
				boost::property_tree::ptree ptree;
				boost::property_tree::ptree& ptree_rootLayer = ptree.add("rootLayer", "");
				ptree_rootLayer.put("<xmlattr>.layerCode", ::GUID2String(pConnectLayerData->GetLayerCode()));
				ptree_rootLayer.put("<xmlattr>.guid", ::GUID2String(pConnectLayerData->GetGUID()));
				ptree_rootLayer.put("<xmlattr>.outputLayerGUID", ::GUID2String(pConnectLayerData->GetOutputLayerGUID()));


				// レイヤーの接続情報を記載する
				boost::property_tree::ptree& ptree_layerConnect = ptree_rootLayer.add("layerConnect", "");
				for(U32 layerNum=0; layerNum<pConnectLayerData->GetLayerCount(); layerNum++)
				{
					Gravisbell::GUID layerGUID;
					if(pConnectLayerData->GetLayerGUIDbyNum(layerNum, layerGUID) != Gravisbell::ErrorCode::ERROR_CODE_NONE)
						continue;

					auto pLayerData = pConnectLayerData->GetLayerDataByGUID(layerGUID);
					if(pLayerData == NULL)
						continue;

					if(!lpAlreadyExportLayerDataGUID.count(pLayerData->GetGUID()))
					{
						// レイヤーが保存済みではない場合、新しく保存
						SaveLayerToXML(*pLayerData, lpAlreadyExportLayerDataGUID, i_layerDirPath, "");
					}

					// GUIDを記載
					boost::property_tree::ptree& ptree_layer = ptree_layerConnect.add("layer", "");
					ptree_layer.put("<xmlattr>.guid", ::GUID2String(layerGUID));
					ptree_layer.put("<xmlattr>.layerCode", ::GUID2String(pLayerData->GetLayerCode()));
					ptree_layer.put("<xmlattr>.layerData", ::GUID2String(pLayerData->GetGUID()));

					// 入力レイヤーを記載
					for(U32 inputNum=0; inputNum<pConnectLayerData->GetInputLayerCount(layerGUID); inputNum++)
					{
						Gravisbell::GUID inputLayerGUID;
						if(pConnectLayerData->GetInputLayerGUIDbyNum(layerGUID, inputNum, inputLayerGUID) != ErrorCode::ERROR_CODE_NONE)
							continue;

						ptree_layer.add("input", ::GUID2String(inputLayerGUID));
					}
					// バイパス入力レイヤーを記載
					for(U32 inputNum=0; inputNum<pConnectLayerData->GetBypassLayerCount(layerGUID); inputNum++)
					{
						Gravisbell::GUID inputLayerGUID;
						if(pConnectLayerData->GetBypassLayerGUIDbyNum(layerGUID, inputNum, inputLayerGUID) != ErrorCode::ERROR_CODE_NONE)
							continue;

						ptree_layer.add("bypass", ::GUID2String(inputLayerGUID));
					}
				}

				// ファイルに保存
				const int indent = 2;
				boost::property_tree::xml_parser::write_xml(
					layerFilePath.generic_string(),
					ptree,
					std::locale(),
					boost::property_tree::xml_parser::xml_writer_make_settings(' ', indent, boost::property_tree::xml_parser::widen<char>("utf-8")));

			}
			else
			{
				// 通常レイヤーの場合

			}

			return ErrorCode::ERROR_CODE_NONE;
		}
	}
	



	/** レイヤーデータをXMLファイルから作成する.
		@param	i_layerDLLManager	レイヤーDLLの管理クラス.
		@param	i_layerDatamanager	レイヤーデータの管理クラス.新規作成されたレイヤーデータはこのクラスに格納される.
		@param	i_layerDirPath		レイヤーデータが格納されているディレクトリパス.
		@param	i_rootLayerFilePath	基準となるレイヤーデータが格納されているXMLファイルパス
		@return	成功した場合レイヤーデータが返る.
		*/
	NetworkParserXML_API INNLayerData* CreateLayerFromXML(const ILayerDLLManager& i_layerDLLManager, ILayerDataManager& io_layerDataManager, const wchar_t i_layerDirPath[], const wchar_t i_rootLayerFilePath[])
	{
		return NULL;
	}

	/** レイヤーデータをXMLファイルに書き出す.
		@param	i_NNLayer			書き出すレイヤーデータ.
		@param	i_layerDirPath		レイヤーデータが格納されているディレクトリパス.
		@param	i_rootLayerFilePath	基準となるレイヤーデータが格納されているXMLファイルパス. 空白が指定された場合、i_layerDirPath内にi_NNLayerのGUIDを名前としたファイルが生成される.
		*/
	NetworkParserXML_API Gravisbell::ErrorCode SaveLayerToXML(INNLayerData& i_NNLayer, const wchar_t i_layerDirPath[], const wchar_t i_rootLayerFilePath[])
	{
		// 出力レイヤーがレイヤー接続型であることを確認
		INNLayerConnectData* pRootLayer = dynamic_cast<INNLayerConnectData*>(&i_NNLayer);
		if(pRootLayer == NULL)
			return ErrorCode::ERROR_CODE_COMMON_NOT_COMPATIBLE;


		// 出力先ディレクトリが存在するか確認
		boost::filesystem::wpath layerDirPath = i_layerDirPath;
		if(!boost::filesystem::exists(layerDirPath))
		{
			return ErrorCode::ERROR_CODE_COMMON_FILE_NOT_FOUND;
		}
		if(!boost::filesystem::is_directory(layerDirPath))
		{
			return ErrorCode::ERROR_CODE_COMMON_FILE_NOT_FOUND;
		}

		std::set<GUID> lpAlreadySaveGUID;

		return SaveLayerToXML(i_NNLayer, lpAlreadySaveGUID, i_layerDirPath, i_rootLayerFilePath);
	}

}	// Parser
}	// NeuralNetwork
}	// Layer
}	// Gravisbell
