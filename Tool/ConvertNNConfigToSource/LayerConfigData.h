//======================================
// レイヤーの設定情報について記載
//======================================
#ifndef __LAYER_CONFIG_DATA__H__
#define __LAYER_CONFIG_DATA__H__

#include<string>

#include<boost/filesystem.hpp>

#include<guiddef.h>
#include"NNLayerConfig.h"

namespace CustomDeepNNLibrary
{
	class LayerConfigData
	{
	private:
		GUID guid;	/**< 識別コード */
		VersionCode version;	/**< バージョン */
		std::wstring default_language;	/**< 基本言語 */

		std::wstring name;	/**< 名前 */
		std::wstring text;	/**< 説明テキスト */

		CustomDeepNNLibrary::INNLayerConfigEx* pStructure;	/**< レイヤー構造定義情報 */
		CustomDeepNNLibrary::INNLayerConfigEx* pLearn;		/**< 学習設定情報 */

	public:
		/** コンストラクタ */
		LayerConfigData();
		/** デストラクタ */
		~LayerConfigData();

	public:
		/** XMLファイルから情報を読み込む.
			@param	configFilePath	読み込むXMLファイルのパス
			@return	成功した場合0が返る. */
		int ReadFromXMLFile(const boost::filesystem::wpath& configFilePath);
		/** C++言語ソースファイルに変換/出力する.
			.h/.cppファイルが生成される.
			@param	exportDirPath	出力先ディレクトリパス
			@param	fileName		出力ファイル名.拡張子は除く.
			@return 成功した場合0が返る. */
		int ConvertToCPPFile(const boost::filesystem::wpath& exportDirPath, const std::wstring& fileName)const;
	};
}


#endif	// __LAYER_CONFIG_DATA__H__
