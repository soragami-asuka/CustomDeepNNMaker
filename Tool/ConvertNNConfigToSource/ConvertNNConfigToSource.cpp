// ConvertNNConfigToSource.cpp : コンソール アプリケーションのエントリ ポイントを定義します。
//

#include "stdafx.h"

#include<boost/filesystem.hpp>

#include"LayerConfigData.h"

int _tmain(int argc, _TCHAR* argv[])
{
#ifdef _DEBUG
	::_CrtSetDbgFlag(_CRTDBG_LEAK_CHECK_DF | _CRTDBG_ALLOC_MEM_DF);
#endif
	
	if(argc < 4)
	{
		printf("引数が少なすぎます\n");
		printf("ConvertNNConfigToSource.exe 設定ファイルパス 出力先ディレクトリパス 出力ファイル名(拡張子無し)\n");
#ifdef _DEBUG
		printf("Press Any Key to Continue\n");
		getc(stdin);
#endif
		return -1;
	}
	boost::filesystem::wpath configFilePath = argv[1];	configFilePath.normalize();
	boost::filesystem::wpath exportDirPath	= argv[2];	exportDirPath.normalize();
	std::wstring fileName = argv[3];

	CustomDeepNNLibrary::LayerConfigData configData;
	if(configData.ReadFromXMLFile(configFilePath) != 0)
		return -1;

	return 0;
}

