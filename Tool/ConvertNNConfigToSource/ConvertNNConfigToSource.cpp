// ConvertNNConfigToSource.cpp : �R���\�[�� �A�v���P�[�V�����̃G���g�� �|�C���g���`���܂��B
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
		printf("���������Ȃ����܂�\n");
		printf("ConvertNNConfigToSource.exe �ݒ�t�@�C���p�X �o�͐�f�B���N�g���p�X �o�̓t�@�C����(�g���q����)\n");
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

