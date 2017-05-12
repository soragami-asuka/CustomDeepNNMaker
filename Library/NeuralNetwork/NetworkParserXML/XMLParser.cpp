//=========================================
// XML�`���ŋL�q���ꂽ�l�b�g���[�N�\������͂���
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
		// ������𕪉�
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
		/** ���C���[�f�[�^��XML�t�@�C���ɏ����o��. */
		Gravisbell::ErrorCode SaveLayerToXML(INNLayerData& i_NNLayer, std::set<Gravisbell::GUID>& lpAlreadyExportLayerDataGUID, const boost::filesystem::wpath& i_layerDirPath, const boost::filesystem::wpath& i_layerFilePath)
		{
			// �o�̓��C���[�����C���[�ڑ��^�ł��邱�Ƃ��m�F
			INNLayerConnectData* pConnectLayerData = dynamic_cast<INNLayerConnectData*>(&i_NNLayer);

			// �o�͐�t�@�C�����������߂��Ԃ��m�F����
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

			// ���g��ۑ��ς݂�
			lpAlreadyExportLayerDataGUID.insert(i_NNLayer.GetGUID());

			if(pConnectLayerData)
			{
				// �ڑ��^�̏ꍇ

				// �c���[���쐬
				boost::property_tree::ptree ptree;
				boost::property_tree::ptree& ptree_rootLayer = ptree.add("rootLayer", "");
				ptree_rootLayer.put("<xmlattr>.layerCode", ::GUID2String(pConnectLayerData->GetLayerCode()));
				ptree_rootLayer.put("<xmlattr>.guid", ::GUID2String(pConnectLayerData->GetGUID()));
				ptree_rootLayer.put("<xmlattr>.outputLayerGUID", ::GUID2String(pConnectLayerData->GetOutputLayerGUID()));


				// ���C���[�̐ڑ������L�ڂ���
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
						// ���C���[���ۑ��ς݂ł͂Ȃ��ꍇ�A�V�����ۑ�
						SaveLayerToXML(*pLayerData, lpAlreadyExportLayerDataGUID, i_layerDirPath, "");
					}

					// GUID���L��
					boost::property_tree::ptree& ptree_layer = ptree_layerConnect.add("layer", "");
					ptree_layer.put("<xmlattr>.guid", ::GUID2String(layerGUID));
					ptree_layer.put("<xmlattr>.layerCode", ::GUID2String(pLayerData->GetLayerCode()));
					ptree_layer.put("<xmlattr>.layerData", ::GUID2String(pLayerData->GetGUID()));

					// ���̓��C���[���L��
					for(U32 inputNum=0; inputNum<pConnectLayerData->GetInputLayerCount(layerGUID); inputNum++)
					{
						Gravisbell::GUID inputLayerGUID;
						if(pConnectLayerData->GetInputLayerGUIDbyNum(layerGUID, inputNum, inputLayerGUID) != ErrorCode::ERROR_CODE_NONE)
							continue;

						ptree_layer.add("input", ::GUID2String(inputLayerGUID));
					}
					// �o�C�p�X���̓��C���[���L��
					for(U32 inputNum=0; inputNum<pConnectLayerData->GetBypassLayerCount(layerGUID); inputNum++)
					{
						Gravisbell::GUID inputLayerGUID;
						if(pConnectLayerData->GetBypassLayerGUIDbyNum(layerGUID, inputNum, inputLayerGUID) != ErrorCode::ERROR_CODE_NONE)
							continue;

						ptree_layer.add("bypass", ::GUID2String(inputLayerGUID));
					}
				}

				// �t�@�C���ɕۑ�
				const int indent = 2;
				boost::property_tree::xml_parser::write_xml(
					layerFilePath.generic_string(),
					ptree,
					std::locale(),
					boost::property_tree::xml_parser::xml_writer_make_settings(' ', indent, boost::property_tree::xml_parser::widen<char>("utf-8")));

			}
			else
			{
				// �ʏ탌�C���[�̏ꍇ

			}

			return ErrorCode::ERROR_CODE_NONE;
		}
	}
	



	/** ���C���[�f�[�^��XML�t�@�C������쐬����.
		@param	i_layerDLLManager	���C���[DLL�̊Ǘ��N���X.
		@param	i_layerDatamanager	���C���[�f�[�^�̊Ǘ��N���X.�V�K�쐬���ꂽ���C���[�f�[�^�͂��̃N���X�Ɋi�[�����.
		@param	i_layerDirPath		���C���[�f�[�^���i�[����Ă���f�B���N�g���p�X.
		@param	i_rootLayerFilePath	��ƂȂ郌�C���[�f�[�^���i�[����Ă���XML�t�@�C���p�X
		@return	���������ꍇ���C���[�f�[�^���Ԃ�.
		*/
	NetworkParserXML_API INNLayerData* CreateLayerFromXML(const ILayerDLLManager& i_layerDLLManager, ILayerDataManager& io_layerDataManager, const wchar_t i_layerDirPath[], const wchar_t i_rootLayerFilePath[])
	{
		return NULL;
	}

	/** ���C���[�f�[�^��XML�t�@�C���ɏ����o��.
		@param	i_NNLayer			�����o�����C���[�f�[�^.
		@param	i_layerDirPath		���C���[�f�[�^���i�[����Ă���f�B���N�g���p�X.
		@param	i_rootLayerFilePath	��ƂȂ郌�C���[�f�[�^���i�[����Ă���XML�t�@�C���p�X. �󔒂��w�肳�ꂽ�ꍇ�Ai_layerDirPath����i_NNLayer��GUID�𖼑O�Ƃ����t�@�C�������������.
		*/
	NetworkParserXML_API Gravisbell::ErrorCode SaveLayerToXML(INNLayerData& i_NNLayer, const wchar_t i_layerDirPath[], const wchar_t i_rootLayerFilePath[])
	{
		// �o�̓��C���[�����C���[�ڑ��^�ł��邱�Ƃ��m�F
		INNLayerConnectData* pRootLayer = dynamic_cast<INNLayerConnectData*>(&i_NNLayer);
		if(pRootLayer == NULL)
			return ErrorCode::ERROR_CODE_COMMON_NOT_COMPATIBLE;


		// �o�͐�f�B���N�g�������݂��邩�m�F
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
