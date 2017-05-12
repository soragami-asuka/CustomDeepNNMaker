//=========================================
// XML形式で記述されたネットワーク構文を解析する
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

	/** レイヤーデータをXMLファイルから作成する.
		@param	i_layerDLLManager	レイヤーDLLの管理クラス.
		@param	i_layerDatamanager	レイヤーデータの管理クラス.新規作成されたレイヤーデータはこのクラスに格納される.
		@param	i_layerDirPath		レイヤーデータが格納されているディレクトリパス.
		@param	i_rootLayerFilePath	基準となるレイヤーデータが格納されているXMLファイルパス
		@return	成功した場合レイヤーデータが返る.
		*/
	extern NetworkParserXML_API INNLayerData* CreateLayerFromXML(const ILayerDLLManager& i_layerDLLManager, ILayerDataManager& io_layerDataManager, const wchar_t i_layerDirPath[], const wchar_t i_rootLayerFilePath[]);

	/** レイヤーデータをXMLファイルに書き出す.
		@param	i_NNLayer			書き出すレイヤーデータ.
		@param	i_layerDirPath		レイヤーデータが格納されているディレクトリパス.
		@param	i_rootLayerFilePath	基準となるレイヤーデータが格納されているXMLファイルパス. 空白が指定された場合、i_layerDirPath内にi_NNLayerのGUIDを名前としたファイルが生成される.
		*/
	extern NetworkParserXML_API Gravisbell::ErrorCode SaveLayerToXML(INNLayerData& i_NNLayer, const wchar_t i_layerDirPath[], const wchar_t i_rootLayerFilePath[] = L"");

}	// Parser
}	// NeuralNetwork
}	// Layer
}	// Gravisbell
