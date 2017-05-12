//=========================================
// XML�`���ŋL�q���ꂽ�l�b�g���[�N�\������͂���
//=========================================
#ifdef NETWORKPARSERXML_EXPORTS
#define NetworkParserXML_API __declspec(dllexport)
#else
#define NetworkParserXML_API __declspec(dllimport)
#endif

#include"Common/ErrorCode.h"
#include"Common/VersionCode.h"

#include"Layer/NeuralNetwork/ILayerDLLManager.h"
#include"Layer/NeuralNetwork/INNLayerData.h"
#include"Layer/NeuralNetwork/ILayerDataManager.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {
namespace Parser {

	/** ���C���[�f�[�^��XML�t�@�C������쐬����.
		@param	i_layerDLLManager	���C���[DLL�̊Ǘ��N���X.
		@param	i_layerDatamanager	���C���[�f�[�^�̊Ǘ��N���X.�V�K�쐬���ꂽ���C���[�f�[�^�͂��̃N���X�Ɋi�[�����.
		@param	i_layerDirPath		���C���[�f�[�^���i�[����Ă���f�B���N�g���p�X.
		@param	i_rootLayerFilePath	��ƂȂ郌�C���[�f�[�^���i�[����Ă���XML�t�@�C���p�X
		@return	���������ꍇ���C���[�f�[�^���Ԃ�.
		*/
	extern NetworkParserXML_API INNLayerData* CreateLayerFromXML(const ILayerDLLManager& i_layerDLLManager, ILayerDataManager& io_layerDataManager, const wchar_t i_layerDirPath[], const wchar_t i_rootLayerFilePath[]);

	/** ���C���[�f�[�^��XML�t�@�C���ɏ����o��.
		@param	i_NNLayer			�����o�����C���[�f�[�^.
		@param	i_layerDirPath		���C���[�f�[�^���i�[����Ă���f�B���N�g���p�X.
		@param	i_rootLayerFilePath	��ƂȂ郌�C���[�f�[�^���i�[����Ă���XML�t�@�C���p�X. �󔒂��w�肳�ꂽ�ꍇ�Ai_layerDirPath����i_NNLayer��GUID�𖼑O�Ƃ����t�@�C�������������.
		*/
	extern NetworkParserXML_API Gravisbell::ErrorCode SaveLayerToXML(INNLayerData& i_NNLayer, const wchar_t i_layerDirPath[], const wchar_t i_rootLayerFilePath[] = L"");

}	// Parser
}	// NeuralNetwork
}	// Layer
}	// Gravisbell
